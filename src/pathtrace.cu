// src/pathtrace.cu
#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>

#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>

#include <vector>
#include <climits>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

//  Feature Toggles 
#define ENABLE_ANTIALIASING       1
#define ENABLE_DEPTH_OF_FIELD     1     // thin lens camera
#define ENABLE_DIRECT_LIGHTING    1     // next-event estimation, 1 sample
#define ENABLE_SORT_BY_MATERIAL   1
#define ENABLE_RUSSIAN_ROULETTE   1
#define POSTPROCESS_ACES          1     // ACES tonemapper in display

#define RR_DEPTH_START            3
#define RR_MIN_P                  0.1f

// Depth of Field defaults (if you don’t add JSON keys)
#ifndef DOF_APERTURE
#define DOF_APERTURE 0.05f // lens radius in world units (0 disables)
#endif
#ifndef DOF_FOCUSDIST
#define DOF_FOCUSDIST 5.0f  // default focus distance from camera
#endif

//  Globals 
static Scene* hst_scene = nullptr;
static GuiDataContainer* hst_guiData = nullptr;

static Geom* dev_geoms = nullptr;
static Material* dev_materials = nullptr;
static glm::vec3* dev_image = nullptr;

static PathSegment* dev_paths = nullptr;
static ShadeableIntersection* dev_intersections = nullptr;

// Sorting scratch
static PathSegment* dev_paths_sorted = nullptr;
static ShadeableIntersection* dev_isects_sorted = nullptr;
static int* dev_sortKeys = nullptr;
static int* dev_indices = nullptr;

// Lights
static int* dev_lightGeomIndices = nullptr;
static int  hst_numLights = 0;
static int  hst_numGeoms = 0;
static int  hst_numMaterials = 0;

// Convenience RNG seed
__host__ __device__ inline thrust::default_random_engine makeSeededRandomEngine(
    int iter, int index, int depth)
{
    unsigned int seed =
        utilhash((unsigned int)(iter + 1) * 9781u
            + (unsigned int)index * 6271u
            + (unsigned int)depth * 0x9e3779b9u);
    thrust::default_random_engine rng(seed);
    rng.discard(index % 97);
    return rng;
}

__host__ __device__ inline glm::vec3 rayPoint(const Ray& r, float t) {
    return r.origin + t * r.direction;
}

//  Light sampling 
__device__ glm::vec3 samplePointOnCube(
    const Geom& g, thrust::default_random_engine& rng,
    glm::vec3& outNormal, float& outArea)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    // Approximate areas from scale (unit cube [-.5,.5]^3)
    glm::vec3 s = g.scale;
    float areaXY = fabsf(s.x * s.y);
    float areaYZ = fabsf(s.y * s.z);
    float areaZX = fabsf(s.z * s.x);
    float faceAreas[6] = { areaYZ, areaYZ, areaZX, areaZX, areaXY, areaXY };
    float total = faceAreas[0] + faceAreas[1] + faceAreas[2] + faceAreas[3] + faceAreas[4] + faceAreas[5];
    outArea = 2.0f * (areaXY + areaYZ + areaZX);

    float xi = u01(rng) * total;
    int face = 0;
    for (; face < 6; ++face) { xi -= faceAreas[face]; if (xi <= 0) break; }
    float u = u01(rng) - 0.5f;
    float v = u01(rng) - 0.5f;

    glm::vec3 pL(0);
    switch (face) {
    case 0: pL = glm::vec3(-0.5f, u, v); outNormal = glm::vec3(-1, 0, 0); break;
    case 1: pL = glm::vec3(0.5f, u, v); outNormal = glm::vec3(1, 0, 0); break;
    case 2: pL = glm::vec3(u, -0.5f, v); outNormal = glm::vec3(0, -1, 0); break;
    case 3: pL = glm::vec3(u, 0.5f, v); outNormal = glm::vec3(0, 1, 0); break;
    case 4: pL = glm::vec3(u, v, -0.5f); outNormal = glm::vec3(0, 0, -1); break;
    default:pL = glm::vec3(u, v, 0.5f); outNormal = glm::vec3(0, 0, 1); break;
    }
    glm::vec3 worldP = multiplyMV(g.transform, glm::vec4(pL, 1.0f));
    outNormal = glm::normalize(multiplyMV(g.invTranspose, glm::vec4(outNormal, 0.0f)));
    return worldP;
}

__device__ glm::vec3 samplePointOnSphere(
    const Geom& g, thrust::default_random_engine& rng,
    glm::vec3& outNormal, float& outArea)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float z = 1.0f - 2.0f * u01(rng);
    float phi = TWO_PI * u01(rng);
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    glm::vec3 pL = glm::vec3(r * cosf(phi), r * sinf(phi), z) * 0.5f; // radius 0.5
    glm::vec3 worldP = multiplyMV(g.transform, glm::vec4(pL, 1.0f));
    glm::vec3 nObj = glm::normalize(pL);
    outNormal = glm::normalize(multiplyMV(g.invTranspose, glm::vec4(nObj, 0.0f)));

    float s = (g.scale.x + g.scale.y + g.scale.z) / 3.0f;
    float rWorld = 0.5f * s;
    outArea = 4.0f * PI * rWorld * rWorld;
    return worldP;
}

__device__ glm::vec3 samplePointOnLight(
    const Geom& light, thrust::default_random_engine& rng,
    glm::vec3& n, float& area)
{
    return (light.type == CUBE)
        ? samplePointOnCube(light, rng, n, area)
        : samplePointOnSphere(light, rng, n, area);
}

//  Kernels 

// Generate camera rays with AA and optional DOF (thin lens)
__global__ void generateCameraRays(
    int iter, int maxDepth, Camera cam, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= cam.resolution.x || y >= cam.resolution.y) return;

    int index = x + y * cam.resolution.x;

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec2 jitter(0);
#if ENABLE_ANTIALIASING
    jitter.x = u01(rng) - 0.5f;
    jitter.y = u01(rng) - 0.5f;
#endif

    float px = (x + 0.5f + jitter.x) * cam.pixelLength.x - 1.0f;
    float py = (y + 0.5f + jitter.y) * cam.pixelLength.y - 1.0f;

    glm::vec3 origin = cam.position;
    glm::vec3 dir = glm::normalize(px * cam.right + py * cam.up + cam.view);

#if ENABLE_DEPTH_OF_FIELD
    float aperture = DOF_APERTURE;
    float focusDist = DOF_FOCUSDIST;
    if (aperture > 0.0f) {
        // concentric disk sample
        float u = 2.f * u01(rng) - 1.f;
        float v = 2.f * u01(rng) - 1.f;
        float r, phi;
        if (u == 0.f && v == 0.f) { r = 0.f; phi = 0.f; }
        else if (fabsf(u) > fabsf(v)) { r = u; phi = (PI / 4.f) * (v / u); }
        else { r = v; phi = (PI / 2.f) - (PI / 4.f) * (u / v); }
        float dx = r * cosf(phi);
        float dy = r * sinf(phi);

        glm::vec3 lensOffset = (dx * cam.right + dy * cam.up) * aperture;
        glm::vec3 focusPoint = origin + dir * focusDist;
        origin += lensOffset;
        dir = glm::normalize(focusPoint - origin);
    }
#endif

    PathSegment ps;
    ps.ray.origin = origin;
    ps.ray.direction = dir;
    ps.color = glm::vec3(1.0f);   // throughput
    ps.pixelIndex = index;
    ps.remainingBounces = maxDepth;

    pathSegments[index] = ps;
}

__global__ void computeIntersections(
    int num_paths, Geom* geoms, int geomsCount,
    PathSegment* pathSegments, ShadeableIntersection* intersections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    const PathSegment& path = pathSegments[idx];

    float tMin = 1e20f;
    int hitMat = -1;
    glm::vec3 n(0), p(0);

    for (int i = 0; i < geomsCount; ++i) {
        const Geom& g = geoms[i];
        glm::vec3 tmpP, tmpN;
        bool outside = true;
        float t = (g.type == SPHERE)
            ? sphereIntersectionTest(g, path.ray, tmpP, tmpN, outside)
            : boxIntersectionTest(g, path.ray, tmpP, tmpN, outside);
        if (t > 0.0f && t < tMin) {
            tMin = t; n = tmpN; p = tmpP; hitMat = g.materialid;
        }
    }

    ShadeableIntersection isect;
    isect.t = (hitMat == -1) ? -1.0f : tMin;
    isect.surfaceNormal = n;
    isect.materialId = hitMat;
    intersections[idx] = isect;
}

// Additive tonemapped accumulation helper
__device__ inline void addToImage(glm::vec3* image, int pixel, const glm::vec3& c) {
    atomicAdd(&image[pixel].x, c.x);
    atomicAdd(&image[pixel].y, c.y);
    atomicAdd(&image[pixel].z, c.z);
}

// Shading kernel: accumulates emission and NEE to dev_image, then spawns new rays
__global__ void shadeMaterials(
    int iter, int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms, int geomsCount,
    int* lightIndices, int numLights,
    glm::vec3* imageOut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& ps = pathSegments[idx];
    ShadeableIntersection& isect = shadeableIntersections[idx];

    if (isect.t < 0.0f) {
        // Miss -> background
        addToImage(imageOut, ps.pixelIndex, ps.color * BACKGROUND_COLOR);
        ps.color = glm::vec3(0);
        ps.remainingBounces = 0;
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, ps.pixelIndex, ps.remainingBounces);
    const Material& m = materials[isect.materialId];

    glm::vec3 x = getPointOnRay(ps.ray, isect.t);
    glm::vec3 n = glm::normalize(isect.surfaceNormal);

    // If light: accumulate and terminate
    if (m.emittance > 0.0f) {
        addToImage(imageOut, ps.pixelIndex, ps.color * (m.color * m.emittance));
        ps.color = glm::vec3(0);
        ps.remainingBounces = 0;
        return;
    }

#if ENABLE_DIRECT_LIGHTING
    // NEE for diffuse interactions only (simple & efficient)
    if (numLights > 0 && m.hasRefractive == 0.0f && m.hasReflective == 0.0f) {
        thrust::uniform_int_distribution<int> uLight(0, numLights - 1);
        int lightIdx = uLight(rng);
        const Geom& light = geoms[lightIndices[lightIdx]];

        glm::vec3 nl;
        float area;
        glm::vec3 y = samplePointOnLight(light, rng, nl, area);
        glm::vec3 wi = glm::normalize(y - x);
        float dist2 = glm::dot(y - x, y - x);

        // Shadow ray (epsilon to avoid self-hits)
        Ray shadowRay;
        shadowRay.origin = x + wi * 0.0002f;
        shadowRay.direction = wi;

        bool occluded = false;
        for (int i = 0; i < geomsCount; ++i) {
            const Geom& g = geoms[i];
            glm::vec3 tmpP, tmpN; bool outside = true;
            float t = (g.type == SPHERE)
                ? sphereIntersectionTest(g, shadowRay, tmpP, tmpN, outside)
                : boxIntersectionTest(g, shadowRay, tmpP, tmpN, outside);
            if (t > 0.0f) {
                float distToHit = glm::length(tmpP - shadowRay.origin);
                float distToLight = sqrtf(dist2);
                // If hit something closer than light (ignore numerical noise)
                if (distToHit + 1e-3f < distToLight && g.materialid != light.materialid) {
                    occluded = true; break;
                }
            }
        }

        if (!occluded) {
            float cosTheta = fmaxf(0.0f, glm::dot(n, wi));
            float cosThetaLight = fmaxf(0.0f, glm::dot(nl, -wi));
            glm::vec3 Le = materials[light.materialid].color * materials[light.materialid].emittance;
            glm::vec3 f = m.color * (1.0f / PI); // Lambertian BRDF
            float G = (cosTheta * cosThetaLight) / fmaxf(1e-6f, dist2);
            float pdfA = 1.0f / fmaxf(1e-6f, area);

            glm::vec3 Ld = Le * f * G / pdfA;  // = Le * f * cos* cosL / dist2 * Area
            // Add with current throughput
            addToImage(imageOut, ps.pixelIndex, ps.color * Ld);
        }
    }
#endif

#if ENABLE_RUSSIAN_ROULETTE
    if (ps.remainingBounces <= RR_DEPTH_START) {
        float p = fmaxf(fmaxf(ps.color.r, ps.color.g), ps.color.b);
        p = fmaxf(p, RR_MIN_P);
        thrust::uniform_real_distribution<float> u01(0, 1);
        if (u01(rng) > p) {
            // terminate with no additional contribution
            ps.color = glm::vec3(0);
            ps.remainingBounces = 0;
            return;
        }
        ps.color /= p; // survival compensation
    }
#endif

    // Scatter new ray according to material (updates throughput & direction)
    scatterRay(ps, x, n, m, rng);
}

//  Display 
static __device__ inline float acesTonemap(float x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return fminf(1.0f, fmaxf(0.0f, (x * (a * x + b)) / (x * (c * x + d) + e)));
}

__global__ void sendImageToPBO(uchar4* pbo, glm::vec3* image, glm::ivec2 resolution, int iter)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;

    int index = x + y * resolution.x;

    // Average over iterations
    glm::vec3 pix = image[index] / (float)iter;

#if POSTPROCESS_ACES
    pix = glm::vec3(acesTonemap(pix.r), acesTonemap(pix.g), acesTonemap(pix.b));
#else
    pix = pix / (glm::vec3(1.0f) + pix);
#endif
    // gamma 2.2
    pix = glm::pow(glm::max(pix, glm::vec3(0.0f)), glm::vec3(1.0f / 2.2f));

    glm::ivec3 color;
    color.x = glm::clamp((int)(pix.x * 255.0f), 0, 255);
    color.y = glm::clamp((int)(pix.y * 255.0f), 0, 255);
    color.z = glm::clamp((int)(pix.z * 255.0f), 0, 255);

    pbo[index].w = 0;
    pbo[index].x = color.x;
    pbo[index].y = color.y;
    pbo[index].z = color.z;
}

//  Host code 
void InitDataContainer(GuiDataContainer* guiData) { hst_guiData = guiData; }

struct PathTerminated {
    __host__ __device__ bool operator()(const PathSegment& ps) const {
        return ps.remainingBounces <= 0;
    }
};
struct MaterialIdToKey {
    __host__ __device__ int operator()(const ShadeableIntersection& s) const {
        return (s.materialId < 0) ? INT_MAX : s.materialId;
    }
};

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = scene->state.camera;
    int pixelcount = cam.resolution.x * cam.resolution.y;

    // Device buffers
    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_paths_sorted, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_isects_sorted, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_sortKeys, pixelcount * sizeof(int));
    cudaMalloc(&dev_indices, pixelcount * sizeof(int));

    // Copy scene data
    hst_numGeoms = (int)scene->geoms.size();
    hst_numMaterials = (int)scene->materials.size();

    cudaMalloc(&dev_geoms, hst_numGeoms * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(),
        hst_numGeoms * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, hst_numMaterials * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(),
        hst_numMaterials * sizeof(Material), cudaMemcpyHostToDevice);

    // Build light index list (host) → device
    std::vector<int> lightIdx;
    lightIdx.reserve(scene->geoms.size());
    for (int i = 0; i < (int)scene->geoms.size(); ++i) {
        int mid = scene->geoms[i].materialid;
        if (scene->materials[mid].emittance > 0.0f) {
            lightIdx.push_back(i);
        }
    }
    hst_numLights = (int)lightIdx.size();
    if (hst_numLights > 0) {
        cudaMalloc(&dev_lightGeomIndices, hst_numLights * sizeof(int));
        cudaMemcpy(dev_lightGeomIndices, lightIdx.data(),
            hst_numLights * sizeof(int), cudaMemcpyHostToDevice);
    }
    else {
        dev_lightGeomIndices = nullptr;
    }
}

void pathtraceFree()
{
    if (dev_image) { cudaFree(dev_image);            dev_image = nullptr; }
    if (dev_paths) { cudaFree(dev_paths);            dev_paths = nullptr; }
    if (dev_intersections) { cudaFree(dev_intersections);    dev_intersections = nullptr; }
    if (dev_paths_sorted) { cudaFree(dev_paths_sorted);     dev_paths_sorted = nullptr; }
    if (dev_isects_sorted) { cudaFree(dev_isects_sorted);    dev_isects_sorted = nullptr; }
    if (dev_sortKeys) { cudaFree(dev_sortKeys);         dev_sortKeys = nullptr; }
    if (dev_indices) { cudaFree(dev_indices);          dev_indices = nullptr; }
    if (dev_geoms) { cudaFree(dev_geoms);            dev_geoms = nullptr; }
    if (dev_materials) { cudaFree(dev_materials);        dev_materials = nullptr; }
    if (dev_lightGeomIndices) { cudaFree(dev_lightGeomIndices); dev_lightGeomIndices = nullptr; }
}

void pathtrace(uchar4* pbo, int /*frame*/, int iteration)
{
    const Camera& cam = hst_scene->state.camera;
    const int W = cam.resolution.x;
    const int H = cam.resolution.y;
    const int pixelcount = W * H;

    // 0) Ray generation
    dim3 block2d(8, 8);
    dim3 grid2d((W + block2d.x - 1) / block2d.x,
        (H + block2d.y - 1) / block2d.y);
    generateCameraRays << <grid2d, block2d >> > (iteration, hst_scene->state.traceDepth, cam, dev_paths);
    cudaDeviceSynchronize();

    // 1) Path tracing loop
    int numPaths = pixelcount;
    int depth = 0;

    while (numPaths > 0 && depth < hst_scene->state.traceDepth) {
        // 1a) Intersections
        dim3 block1d(128);
        dim3 grid1d((numPaths + block1d.x - 1) / block1d.x);
        computeIntersections << <grid1d, block1d >> > (numPaths, dev_geoms, hst_numGeoms, dev_paths, dev_intersections);
        cudaDeviceSynchronize();

#if ENABLE_SORT_BY_MATERIAL
        // Keys & indices
        thrust::device_ptr<ShadeableIntersection> isBeg(dev_intersections);
        thrust::device_ptr<int> keyBeg(dev_sortKeys);
        thrust::device_ptr<int> idxBeg(dev_indices);
        thrust::sequence(idxBeg, idxBeg + numPaths, 0);
        thrust::transform(isBeg, isBeg + numPaths, keyBeg, MaterialIdToKey());
        thrust::sort_by_key(keyBeg, keyBeg + numPaths, idxBeg);

        // Gather into sorted buffers
        thrust::device_ptr<PathSegment> pBeg(dev_paths);
        thrust::device_ptr<PathSegment> pOut(dev_paths_sorted);
        thrust::gather(idxBeg, idxBeg + numPaths, pBeg, pOut);

        thrust::device_ptr<ShadeableIntersection> isOut(dev_isects_sorted);
        thrust::gather(idxBeg, idxBeg + numPaths, isBeg, isOut);

        // Swap buffers
        { PathSegment* tmp = dev_paths; dev_paths = dev_paths_sorted; dev_paths_sorted = tmp; }
        { ShadeableIntersection* tmp = dev_intersections; dev_intersections = dev_isects_sorted; dev_isects_sorted = tmp; }
#endif

        // 1b) Shade (accumulate NEE + emission to dev_image)
        shadeMaterials << <grid1d, block1d >> > (
            iteration, numPaths,
            dev_intersections, dev_paths, dev_materials,
            dev_geoms, hst_numGeoms,
            dev_lightGeomIndices, hst_numLights,
            dev_image);
        cudaDeviceSynchronize();

        // 1c) Stream compaction: remove dead paths
        thrust::device_ptr<PathSegment> pBeg2(dev_paths);
        thrust::device_ptr<PathSegment> pEnd2 = pBeg2 + numPaths;
        auto newEnd = thrust::remove_if(thrust::device, pBeg2, pEnd2, PathTerminated());
        numPaths = (int)(newEnd - pBeg2);

        depth++;
        if (hst_guiData) { hst_guiData->TracedDepth = depth; }
    }

    // 2) Display (average by iteration)
    sendImageToPBO << <grid2d, block2d >> > (pbo, dev_image, cam.resolution, iteration);
    cudaDeviceSynchronize();

    // 3) Copy image back to host for saving
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}
