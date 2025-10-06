#include "pathtrace.h"
#include "sceneStructs.h"
#include "scene.h"
#include "intersections.h"
#include "interactions.h"
#include "utilities.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <glm/gtx/norm.hpp>
#include <vector>
#include <climits>

//  Feature toggles & params (no struct changes) 
#define ENABLE_DOF               1   // Depth of Field ()
#define DOF_APERTURE             0.12f
#define DOF_FOCUS_DIST           -1.f    // < 0 -> auto: |lookAt-position| - 

#define ENABLE_CAMERA_MBLUR      1   // Camera position lerp
#define ENABLE_NEE               1   // Next-Event Estimation (direct lighting)
#define ENABLE_ACES_TONEMAP      1   // Post FX
#define ENABLE_SORT_BY_MATERIAL  1   // Group by BSDF to reduce divergence (wavefront-lite)

//  Globals 
static int        gW = 0, gH = 0;
static int        gTraceDepth = 5;
static Geom* dev_geoms = nullptr;
static Material* dev_mats = nullptr;
static glm::vec3* dev_accum = nullptr;
static PathSegment* dev_paths = nullptr;
static ShadeableIntersection* dev_isects = nullptr;
static int* dev_matKeys = nullptr;       // material sort keys

static int        hNumGeoms = 0, hNumMats = 0;

// emissive object list (indices into geoms)
static int* dev_lightIdx = nullptr;
static int        hNumLights = 0;

static RenderState* hState = nullptr;
static glm::vec3* hImage = nullptr;
static GuiDataContainer* hGui = nullptr;

//  Utilities 
__host__ __device__ inline unsigned int utilhash_local(unsigned int a)
{
    // tiny integer hash; good enough for scrambling/QMC CP-rot
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__device__ inline thrust::default_random_engine makeRNG(int iter, int pixel, int depth) {
    return thrust::default_random_engine(utilhash_local(iter * 9781u + pixel * 9176u + depth * 26699u));
}

// Radical inverse base-2 (Van der Corput) + Hammersley 2D
__device__ inline float radicalInverse_VdC(unsigned int bits) {
    bits = __brev(bits);
    return (float)bits * 2.3283064365386963e-10f; // / 2^32
}
__device__ inline glm::vec2 hammersley2D(unsigned int i, unsigned int N) {
    return glm::vec2((float)i / (float)N, radicalInverse_VdC(i));
}

__device__ inline float fractf(float x) { return x - floorf(x); }
__device__ inline glm::vec2 fract2(glm::vec2 v) { return glm::vec2(fractf(v.x), fractf(v.y)); }

// Per-pixel Cranley–Patterson rotation for LDS (scrambles but stays in [0,1))
__device__ inline glm::vec2 cpRotation2D(int pixel) {
    unsigned int h0 = utilhash_local((unsigned int)pixel * 0x9E3779B9u + 0x85ebca6bu);
    unsigned int h1 = utilhash_local((unsigned int)pixel * 0x85ebca6bu + 0xc2b2ae35u);
    return glm::vec2(radicalInverse_VdC(h0), radicalInverse_VdC(h1));
}

// Concentric disk (for thin lens)
__device__ inline void concentricSampleDisk(float u1, float u2, float& dx, float& dy) {
    float sx = 2.f * u1 - 1.f;
    float sy = 2.f * u2 - 1.f;
    if (sx == 0 && sy == 0) { dx = dy = 0; return; }
    float r, theta;
    if (fabsf(sx) > fabsf(sy)) { r = sx; theta = (PI / 4.f) * (sy / sx); }
    else { r = sy; theta = (PI / 2.f) - (PI / 4.f) * (sx / sy); }
    dx = r * cosf(theta);
    dy = r * sinf(theta);
}

// ACES filmic (Narkowicz 2016 fit)
__device__ inline glm::vec3 ACESFilm(glm::vec3 x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.f, 1.f);
}

//  Device helpers 
__device__ inline void samplePointOnUnitSphere(float u1, float u2, glm::vec3& p, glm::vec3& n) {
    float z = 1.f - 2.f * u1;
    float r = sqrtf(fmaxf(0.f, 1.f - z * z));
    float phi = TWO_PI * u2;
    p = 0.5f * glm::vec3(r * cosf(phi), r * sinf(phi), z);
    n = glm::normalize(p * 2.f);
}

__device__ inline void samplePointOnUnitCube(float u1, float u2, glm::vec3& p, glm::vec3& n) {
    int face = (int)floorf(u1 * 6.f);
    float su = 2.f * (u2)-1.f;

    // simple second dim variation (not area-correct; OK for NEE point picking)
    float sv = 2.f * (radicalInverse_VdC((unsigned)(u1 * 65535u) + 7u)) - 1.f;
    switch (face) {
    case 0: p = glm::vec3(0.5f, su * 0.5f, sv * 0.5f); n = glm::vec3(1, 0, 0);  break;
    case 1: p = glm::vec3(-0.5f, su * 0.5f, sv * 0.5f); n = glm::vec3(-1, 0, 0); break;
    case 2: p = glm::vec3(su * 0.5f, 0.5f, sv * 0.5f); n = glm::vec3(0, 1, 0);  break;
    case 3: p = glm::vec3(su * 0.5f, -0.5f, sv * 0.5f); n = glm::vec3(0, -1, 0); break;
    case 4: p = glm::vec3(su * 0.5f, sv * 0.5f, 0.5f); n = glm::vec3(0, 0, 1);  break;
    default:p = glm::vec3(su * 0.5f, sv * 0.5f, -0.5f); n = glm::vec3(0, 0, -1); break;
    }
}

__device__ inline glm::vec3 multiplyMV_dev(const glm::mat4& m, const glm::vec4& v) {
    return glm::vec3(m * v);
}
__device__ inline glm::vec3 xformPoint(const Geom& g, const glm::vec3& p) {
    return multiplyMV_dev(g.transform, glm::vec4(p, 1.f));
}
__device__ inline glm::vec3 xformNormal(const Geom& g, const glm::vec3& n) {
    return glm::normalize(multiplyMV_dev(g.invTranspose, glm::vec4(n, 0.f)));
}

__device__ inline bool occluded(const Ray& r, float tMax, const Geom* geoms, int nGeoms) {
    glm::vec3 tmpP, tmpN; bool tmpOut = true;
    for (int i = 0; i < nGeoms; ++i) {
        float t = (geoms[i].type == SPHERE)
            ? sphereIntersectionTest(geoms[i], r, tmpP, tmpN, tmpOut)
            : boxIntersectionTest(geoms[i], r, tmpP, tmpN, tmpOut);
        if (t > 0.f && t < tMax) return true;
    }
    return false;
}

//  Kernels 
__global__ void kernGenerateCameraRays(
    int w, int h, Camera cam, int iter, PathSegment* paths)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = x + y * w;

    // Low-discrepancy jitter + CP rotation per pixel (better convergence).
    glm::vec2 jitter = fract2(hammersley2D((unsigned)(iter - 1), 1024u) + cpRotation2D(idx));
    float u = (x + jitter.x) * cam.pixelLength.x - 1.f;
    float v = (y + jitter.y) * cam.pixelLength.y - 1.f;

    // Camera shutter blur between two positions
    glm::vec3 camPos = cam.position;

#if ENABLE_CAMERA_MBLUR
    thrust::default_random_engine rng = makeRNG(iter, idx, 0);
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    float t = u01(rng);
    glm::vec3 p0 = cam.position + glm::vec3(-0.02f, 0.0f, 0.00f);
    glm::vec3 p1 = cam.position + glm::vec3(0.02f, 0.0f, 0.00f);
    camPos = (1.f - t) * p0 + t * p1;
#endif

    glm::vec3 dir = glm::normalize(u * cam.right + v * cam.up + cam.view);
    glm::vec3 rayOrig = camPos;

#if ENABLE_DOF
    if (DOF_APERTURE > 0.f) {
        float focusDist = (DOF_FOCUS_DIST > 0.f)
            ? DOF_FOCUS_DIST
            : glm::length(cam.lookAt - cam.position);
        float lensRadius = 0.5f * DOF_APERTURE;
        glm::vec2 lxi = fract2(hammersley2D((unsigned)(iter + 131), 1024u) + cpRotation2D(idx));
        float dx, dy;
        concentricSampleDisk(lxi.x, lxi.y, dx, dy);
        glm::vec3 pLens = camPos + dx * lensRadius * cam.right + dy * lensRadius * cam.up;
        glm::vec3 pFocus = camPos + dir * focusDist;
        dir = glm::normalize(pFocus - pLens);
        rayOrig = pLens;
    }
#endif

    PathSegment ps;
    ps.ray.origin = rayOrig;
    ps.ray.direction = dir;
    ps.color = glm::vec3(1.f);
    ps.pixelIndex = idx;
    ps.remainingBounces = INT_MAX; // will clamp next
    paths[idx] = ps;
}

__global__ void kernClampDepth(PathSegment* paths, int N, int depth) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= N) return;
    paths[i].remainingBounces = depth;
}

__global__ void kernComputeIntersections(
    int N, const PathSegment* paths, const Geom* geoms, int nGeoms,
    ShadeableIntersection* isects)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= N) return;

    const Ray r = paths[i].ray;
    float tMin = 1e20f;
    int   hitMat = -1;
    glm::vec3 hitN(0), hitP(0);
    bool outside = true;

    for (int g = 0; g < nGeoms; ++g) {
        glm::vec3 p, n; bool out = true;
        float t = (geoms[g].type == SPHERE)
            ? sphereIntersectionTest(geoms[g], r, p, n, out)
            : boxIntersectionTest(geoms[g], r, p, n, out);
        if (t > 0.f && t < tMin) {
            tMin = t; hitN = n; hitP = p; hitMat = geoms[g].materialid; outside = out;
        }
    }

    ShadeableIntersection si;
    si.t = (hitMat >= 0) ? tMin : -1.f;
    si.surfaceNormal = hitN;
    si.materialId = hitMat;
    isects[i] = si;
}

__global__ void kernMakeMaterialKeys(int N, const ShadeableIntersection* isects, int* keys) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= N) return;
    int id = isects[i].materialId;
    // Misses (id < 0) go to the end to keep real hits tightly grouped
    keys[i] = (id >= 0) ? id : INT_MAX;
}

__global__ void kernShadeScatterAndNEE(
    int iter, int depth, int N, PathSegment* paths, const ShadeableIntersection* isects,
    const Material* mats, const Geom* geoms, int nGeoms,
    const int* lightIdx, int nLights, glm::vec3* accum)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= N) return;

    PathSegment& ps = paths[i];
    if (ps.remainingBounces <= 0) return;

    const ShadeableIntersection& si = isects[i];
    int pixel = ps.pixelIndex;

    thrust::default_random_engine rng = makeRNG(iter, pixel, depth);

    // Miss => background
    if (si.t < 0.f) { ps.remainingBounces = 0; return; }

    const Material m = mats[si.materialId];
    glm::vec3 x = ps.ray.origin + glm::normalize(ps.ray.direction) * si.t;
    glm::vec3 n = glm::normalize(si.surfaceNormal);

    // Hit an emissive: add light and terminate
    if (m.emittance > 0.f) {
        accum[pixel] += ps.color * (m.color * m.emittance);
        ps.remainingBounces = 0;
        return;
    }

    // Russian roulette after a few bounces
    if (depth >= 3) {
        thrust::uniform_real_distribution<float> u01(0.f, 1.f);
        // survival prob based on brightness of the path throughput
        float p = fminf(0.95f, fmaxf(0.05f, fmax(fmax(ps.color.r, ps.color.g), ps.color.b)));
        if (u01(rng) > p) { ps.remainingBounces = 0; return; }
        ps.color /= p; // keep estimator unbiased
    }

#if ENABLE_NEE
    if (nLights > 0) {
        // Use hashed Van der Corput for (u1,u2) instead of RNG to reduce noise
        float u1 = radicalInverse_VdC(utilhash_local(pixel * 1973u + depth * 9277u + iter * 26699u));
        float u2 = radicalInverse_VdC(utilhash_local(pixel * 1259u + depth * 7411u + iter * 31847u + 1u));

        int Lpick = (int)floorf(u1 * (float)nLights);
        Lpick = min(Lpick, nLights - 1);
        const Geom L = geoms[lightIdx[Lpick]];
        const Material mL = mats[L.materialid];

        glm::vec3 pL, nL;
        if (L.type == SPHERE) samplePointOnUnitSphere(u1, u2, pL, nL);
        else                  samplePointOnUnitCube(u1, u2, pL, nL);
        pL = xformPoint(L, pL);
        nL = xformNormal(L, nL);

        glm::vec3 wi = glm::normalize(pL - x);
        float dist = glm::length(pL - x);
        float cosNo = fmaxf(0.f, glm::dot(n, wi));
        float cosLi = fmaxf(0.f, glm::dot(nL, -wi));
        if (cosNo > 0.f && cosLi > 0.f) {
            Ray shadow; shadow.origin = x + n * 1e-4f; shadow.direction = wi;
            bool blocked = occluded(shadow, dist * 0.999f, geoms, nGeoms);
            if (!blocked) {
                glm::vec3 f = m.color * (1.f / PI);   // Lambertian BRDF
                glm::vec3 Le = mL.color * mL.emittance;
                glm::vec3 contrib = Le * f * cosNo * (float)nLights; // balance heuristic (1/pSel)
                accum[pixel] += ps.color * contrib;
            }
        }
    }
#endif

    // Scatter next direction (Lambert/mirror/dielectric)
    scatterRay(ps, x, n, m, rng);
}

struct IsTerminated {
    __host__ __device__ bool operator()(const PathSegment& p) const { return p.remainingBounces <= 0; }
};

__global__ void kernToPBO(uchar4* pbo, int w, int h, const glm::vec3* accum, int iter)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = x + y * w;

    glm::vec3 c = accum[idx] / fmaxf(1.f, (float)iter);
#if ENABLE_ACES_TONEMAP
    c = ACESFilm(c);   // ACES filmic fit
#endif
    c = glm::pow(glm::clamp(c, glm::vec3(0), glm::vec3(1)), glm::vec3(1.f / 2.2f));
    pbo[idx] = make_uchar4(
        (unsigned char)(255.f * c.r),
        (unsigned char)(255.f * c.g),
        (unsigned char)(255.f * c.b), 255);
}

//  Host API 
void InitDataContainer(GuiDataContainer* guiData) { hGui = guiData; }

void pathtraceInit(Scene* scene)
{
    hState = &scene->state;
    hImage = scene->state.image.data();

    const Camera& cam = hState->camera;
    gW = cam.resolution.x;
    gH = cam.resolution.y;
    gTraceDepth = hState->traceDepth;

    hNumGeoms = (int)scene->geoms.size();
    hNumMats = (int)scene->materials.size();

    const int N = gW * gH;

    cudaMalloc(&dev_geoms, sizeof(Geom) * hNumGeoms);
    cudaMalloc(&dev_mats, sizeof(Material) * hNumMats);
    cudaMalloc(&dev_accum, sizeof(glm::vec3) * N);
    cudaMalloc(&dev_paths, sizeof(PathSegment) * N);
    cudaMalloc(&dev_isects, sizeof(ShadeableIntersection) * N);
    cudaMalloc(&dev_matKeys, sizeof(int) * N);

    cudaMemcpy(dev_geoms, scene->geoms.data(), sizeof(Geom) * hNumGeoms, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mats, scene->materials.data(), sizeof(Material) * hNumMats, cudaMemcpyHostToDevice);

    // Collect emissive geometry indices (host side)
    std::vector<int> lightIdx;
    for (int i = 0; i < (int)scene->geoms.size(); ++i) {
        const Material& m = scene->materials[scene->geoms[i].materialid];
        if (m.emittance > 0.f) lightIdx.push_back(i);
    }
    hNumLights = (int)lightIdx.size();
    if (hNumLights > 0) {
        cudaMalloc(&dev_lightIdx, sizeof(int) * hNumLights);
        cudaMemcpy(dev_lightIdx, lightIdx.data(), sizeof(int) * hNumLights, cudaMemcpyHostToDevice);
    }
    else {
        dev_lightIdx = nullptr;
    }

    // Start from whatever image host already has (for checkpoint resume)
    cudaMemcpy(dev_accum, hImage, sizeof(glm::vec3) * N, cudaMemcpyHostToDevice);
}

void pathtraceFree()
{
    if (dev_geoms)     cudaFree(dev_geoms), dev_geoms = nullptr;
    if (dev_mats)      cudaFree(dev_mats), dev_mats = nullptr;
    if (dev_accum)     cudaFree(dev_accum), dev_accum = nullptr;
    if (dev_paths)     cudaFree(dev_paths), dev_paths = nullptr;
    if (dev_isects)    cudaFree(dev_isects), dev_isects = nullptr;
    if (dev_matKeys)   cudaFree(dev_matKeys), dev_matKeys = nullptr;
    if (dev_lightIdx)  cudaFree(dev_lightIdx), dev_lightIdx = nullptr;

    hState = nullptr;
    hImage = nullptr;
}

void pathtrace(uchar4* pbo, int /*frame*/, int iteration)
{
    const Camera cam = hState->camera;
    const int W = gW, H = gH, N = W * H;

    dim3 block2D(8, 8);
    dim3 grid2D((W + block2D.x - 1) / block2D.x,
        (H + block2D.y - 1) / block2D.y);
    dim3 block1D(256);
    dim3 grid1D((N + block1D.x - 1) / block1D.x);

    // Generate primary rays (AA, DoF, camera MB)
    kernGenerateCameraRays << <grid2D, block2D >> > (W, H, cam, iteration, dev_paths);
    kernClampDepth << <grid1D, block1D >> > (dev_paths, N, gTraceDepth);

    // Trace wave (simple megakernel loop)
    int nPaths = N;
    int depth = 0;
    while (nPaths > 0 && depth < gTraceDepth) {
        int blocks = (nPaths + block1D.x - 1) / block1D.x;

        kernComputeIntersections << <blocks, block1D >> > (nPaths, dev_paths, dev_geoms, hNumGeoms, dev_isects);

#if ENABLE_SORT_BY_MATERIAL
        // Sort by material to reduce divergence in shading
        kernMakeMaterialKeys << <blocks, block1D >> > (nPaths, dev_isects, dev_matKeys);

        thrust::device_ptr<int>   kBeg(dev_matKeys);
        thrust::device_ptr<int>   kEnd = kBeg + nPaths;
        auto zBeg = thrust::make_zip_iterator(
            thrust::make_tuple(thrust::device_pointer_cast(dev_isects),
                thrust::device_pointer_cast(dev_paths)));
        auto zEnd = zBeg + nPaths;

        thrust::sort_by_key(kBeg, kEnd, zBeg);
#endif

        kernShadeScatterAndNEE << <blocks, block1D >> > (
            iteration, depth, nPaths, dev_paths, dev_isects,
            dev_mats, dev_geoms, hNumGeoms,
            dev_lightIdx, hNumLights, dev_accum);

        // Stream compact dead paths (Thrust)
        thrust::device_ptr<PathSegment> pBeg(dev_paths);
        thrust::device_ptr<PathSegment> pEnd = pBeg + nPaths;
        pEnd = thrust::remove_if(pBeg, pEnd, IsTerminated());
        nPaths = (int)(pEnd - pBeg);

        depth++;
        if (hGui) hGui->TracedDepth = depth;
    }

    // Display
    kernToPBO << <grid2D, block2D >> > (pbo, W, H, dev_accum, iteration);

    // Keep accumulator updated on host (for saves/checkpoints)
    cudaMemcpy(hImage, dev_accum, sizeof(glm::vec3) * N, cudaMemcpyDeviceToHost);
}
