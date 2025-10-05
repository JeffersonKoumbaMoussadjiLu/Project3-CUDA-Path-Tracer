// src/interactions.cu
#include "interactions.h"
#include "intersections.h"
#include "utilities.h" 


#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/compatibility.hpp>

#include <algorithm>

//  helpers 
__host__ __device__ static inline float clamp01(float x) {
    return fminf(1.0f, fmaxf(0.0f, x));
}

__host__ __device__ static inline float schlickFresnel(float cosTheta, float etaI, float etaT) {
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 = r0 * r0;
    float m = 1.0f - clamp01(cosTheta);
    return r0 + (1.0f - r0) * m * m * m * m * m;
}

// Build an ONB around n (Frisvad 2012-ish)
__host__ __device__ static inline void makeONB(const glm::vec3& n, glm::vec3& t, glm::vec3& b) {
    if (fabsf(n.z) < 0.999f) {
        t = glm::normalize(glm::cross(glm::vec3(0, 0, 1), n));
    }
    else {
        t = glm::normalize(glm::cross(glm::vec3(0, 1, 0), n));
    }
    b = glm::cross(n, t);
}

//  cosine-weighted hemisphere 
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    // Shirley-Chiu concentric disk -> cosine hemisphere
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    float u = 2.f * u01(rng) - 1.f;
    float v = 2.f * u01(rng) - 1.f;

    float r, phi;
    if (u == 0.f && v == 0.f) {
        r = 0.f; phi = 0.f;
    }
    else {
        if (fabsf(u) > fabsf(v)) { r = u; phi = (PI / 4.f) * (v / u); }
        else { r = v; phi = (PI / 2.f) - (PI / 4.f) * (u / v); }
    }
    float dx = r * cosf(phi);
    float dy = r * sinf(phi);

    float z = sqrtf(fmaxf(0.f, 1.f - dx * dx - dy * dy));
    glm::vec3 t, b;
    makeONB(glm::normalize(normal), t, b);
    return glm::normalize(dx * t + dy * b + z * glm::normalize(normal));
}

//  scatter 
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 x,
    glm::vec3 n,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 wo = pathSegment.ray.direction;        // outgoing from x
    n = glm::normalize(n);
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);

    const bool isDiffuse = (m.hasReflective <= 0.f && m.hasRefractive <= 0.f);
    const bool isMirror = (m.hasReflective > 0.f && m.hasRefractive <= 0.f);
    const bool isDielectric = (m.hasRefractive > 0.f);

    if (isDiffuse) {
        // Cosine-weighted sampling → throughput *= albedo (cos/pdf cancels)
        glm::vec3 wi = calculateRandomDirectionInHemisphere(n, rng);
        pathSegment.ray.origin = x + wi * 0.0002f;
        pathSegment.ray.direction = wi;
        pathSegment.color *= m.color;
    }
    else if (isMirror) {
        // Perfect specular reflection
        glm::vec3 N = (glm::dot(wo, n) < 0.f) ? n : -n;
        glm::vec3 wi = glm::reflect(wo, N);
        pathSegment.ray.origin = x + wi * 0.0002f;
        pathSegment.ray.direction = glm::normalize(wi);
        pathSegment.color *= m.color;  // mirror tint, if any
    }
    else if (isDielectric) {
        // Dielectric with Schlick Fresnel
        glm::vec3 N = (glm::dot(wo, n) < 0.f) ? n : -n;
        float etaI = 1.0f;
        float etaT = (m.indexOfRefraction > 0.f ? m.indexOfRefraction : 1.5f);
        bool entering = (glm::dot(wo, n) < 0.f);
        float eta = entering ? (etaI / etaT) : (etaT / etaI);

        float cosThetaI = fabsf(glm::dot(-wo, N));
        float Fr = schlickFresnel(cosThetaI, etaI, etaT);

        glm::vec3 dirT = glm::refract(wo, N, eta);

        bool tir = (glm::dot(dirT, dirT) < 1e-8f);
        if (tir || u01(rng) < Fr) {
            // Reflect
            glm::vec3 wi = glm::reflect(wo, N);
            pathSegment.ray.origin = x + wi * 0.0002f;
            pathSegment.ray.direction = glm::normalize(wi);
            pathSegment.color *= m.color;
        }
        else {
            // Refract
            glm::vec3 wi = glm::normalize(dirT);
            pathSegment.ray.origin = x + wi * 0.0002f;
            pathSegment.ray.direction = wi;

            // Scale throughput for transmission:
            // Many renderers include (eta^2) factor for radiance scaling across IOR boundary.
            // We keep it simple & energy-reasonable here.
            pathSegment.color *= m.color;
        }
    }

    pathSegment.remainingBounces -= 1;
}
