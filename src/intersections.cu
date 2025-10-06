// src/intersections.cu
#include "intersections.h"
#include <glm/gtc/matrix_inverse.hpp>

// Transformations
__host__ __device__ static inline glm::vec3 xformPoint(const glm::mat4& m, const glm::vec3& p) {
    return glm::vec3(m * glm::vec4(p, 1.f));
}

__host__ __device__ static inline glm::vec3 xformVector(const glm::mat4& m, const glm::vec3& v) {
    return glm::vec3(m * glm::vec4(v, 0.f));
}

// Sphere intersection test
__host__ __device__ float sphereIntersectionTest(
    Geom sphere, Ray r, glm::vec3& outP, glm::vec3& outN, bool& outside)
{
    // Ray -> object space
    glm::vec3 ro = xformPoint(sphere.inverseTransform, r.origin);
    glm::vec3 rd = glm::normalize(xformVector(sphere.inverseTransform, r.direction));

    // unit sphere radius 0.5 at origin
    float a = glm::dot(rd, rd);
    float b = 2.f * glm::dot(rd, ro);
    float c = glm::dot(ro, ro) - 0.25f;

    float disc = b * b - 4.f * a * c;
    if (disc < 0.f) return -1.f;

    float sdisc = sqrtf(disc);
    float t0 = (-b - sdisc) / (2.f * a);
    float t1 = (-b + sdisc) / (2.f * a);
    float t = (t0 > 0.f) ? t0 : ((t1 > 0.f) ? t1 : -1.f);
    if (t <= 0.f) return -1.f;

    glm::vec3 pObj = ro + t * rd;
    glm::vec3 nObj = glm::normalize(pObj);

    glm::vec3 pW = xformPoint(sphere.transform, pObj);
    glm::vec3 nW = glm::normalize(xformVector(sphere.invTranspose, nObj));

    outP = pW;
    outN = nW;
    outside = glm::dot(glm::normalize(r.direction), nW) < 0.f;
    return glm::length(pW - r.origin);
}

__host__ __device__ float boxIntersectionTest(
    Geom box, Ray r, glm::vec3& outP, glm::vec3& outN, bool& outside)
{
    // Ray -> object space
    glm::vec3 ro = xformPoint(box.inverseTransform, r.origin);
    glm::vec3 rd = glm::normalize(xformVector(box.inverseTransform, r.direction));

    // slabs [-0.5, 0.5]^3
    glm::vec3 t1 = (-0.5f - ro) / rd;
    glm::vec3 t2 = (0.5f - ro) / rd;

    glm::vec3 tminv = glm::min(t1, t2);
    glm::vec3 tmaxv = glm::max(t1, t2);
    float tmin = fmaxf(fmaxf(tminv.x, tminv.y), tminv.z);
    float tmax = fminf(fminf(tmaxv.x, tmaxv.y), tmaxv.z);

    if (tmax < 0.f || tmin > tmax) return -1.f;

    float t = (tmin > 0.f) ? tmin : tmax;
    glm::vec3 pObj = ro + t * rd;

	// compute normal in object space
    glm::vec3 nObj(0);
    glm::vec3 ap = glm::abs(pObj);
    if (ap.x > ap.y && ap.x > ap.z) nObj = glm::vec3((pObj.x > 0.f) ? 1.f : -1.f, 0, 0);
    else if (ap.y > ap.z)                nObj = glm::vec3(0, (pObj.y > 0.f) ? 1.f : -1.f, 0);
    else                                 nObj = glm::vec3(0, 0, (pObj.z > 0.f) ? 1.f : -1.f);

    glm::vec3 pW = xformPoint(box.transform, pObj);
    glm::vec3 nW = glm::normalize(xformVector(box.invTranspose, nObj));

    outP = pW;
    outN = nW;
    outside = glm::dot(glm::normalize(r.direction), nW) < 0.f;
    return glm::length(pW - r.origin);
}