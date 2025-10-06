#include "interactions.h"
#include "utilities.h"
#include <glm/gtx/norm.hpp>

// Build an orthonormal basis given normal n
__host__ __device__ static inline void makeONB(const glm::vec3& n, glm::vec3& s, glm::vec3& t) {
    if (fabsf(n.x) > fabsf(n.z)) s = glm::normalize(glm::vec3(-n.y, n.x, 0));
    else                         s = glm::normalize(glm::vec3(0, -n.z, n.y));
    t = glm::cross(n, s);
}

// Cosine-weighted random direction in hemisphere
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng)
{
    // Cosine-weighted sample on the unit hemisphere (PBRT Diffuse).
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    float u = u01(rng);
    float v = u01(rng);
    float r = sqrtf(u);
    float phi = TWO_PI * v;

	// local
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.f, 1.f - u));

	// world
    glm::vec3 n = glm::normalize(normal), s, t;
    makeONB(n, s, t);
    return glm::normalize(x * s + y * t + z * n);
}

// Schlick Fresnel for dielectrics (n_i -> n_t)
// Returns the reflection coefficient
__host__ __device__ static inline float fresnelSchlick(float cosTheta, float etaI, float etaT) {
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 *= r0;
    float m = 1.f - cosTheta;
    return r0 + (1.f - r0) * m * m * m * m * m;
}

// Scatter ray upon intersection
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{   
	// normal must be normalized
    glm::vec3 n = glm::normalize(normal);
    glm::vec3 wo = -glm::normalize(pathSegment.ray.direction);

    if (m.emittance > 0.f) { // light: terminate
        pathSegment.remainingBounces = 0;
        return;
    }

	// rng
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);

    // Dielectric (refraction)
	// Handle reflection and refraction with Fresnel
    if (m.hasRefractive > 0.f) {
        float etaI = 1.f;
        float etaT = fmaxf(1e-6f, m.indexOfRefraction);
        float cosI = glm::dot(wo, n);
        bool entering = (cosI > 0.f);
        glm::vec3 nl = entering ? n : -n;
        float eta = entering ? (etaI / etaT) : (etaT / etaI);
        float Fr = fresnelSchlick(fabsf(cosI), etaI, etaT); // Schlick

		// Russian roulette
		// reflect with probability Fr, else transmit
		// Note that we don't need to divide by pdf here
		// because we multiply the throughput by the albedo (m.color)
        if (u01(rng) < Fr) {
            // reflect
            glm::vec3 wr = glm::reflect(-wo, nl);
            pathSegment.ray.origin = intersect + nl * 1e-4f;
            pathSegment.ray.direction = glm::normalize(wr);
            pathSegment.color *= m.color;
        }
        
        else {
            // transmit
            glm::vec3 wt = glm::refract(-wo, nl, eta);
            if (glm::length2(wt) < 1e-10f) {
                // TIR -> reflect
                glm::vec3 wr = glm::reflect(-wo, nl);
                pathSegment.ray.origin = intersect + nl * 1e-4f;
                pathSegment.ray.direction = glm::normalize(wr);
            }

			// no TIR -> refract
            else {
                pathSegment.ray.origin = intersect - nl * 1e-4f;
                pathSegment.ray.direction = glm::normalize(wt);
            }
			pathSegment.color *= m.color; // No Fresnel division since we use Russian roulette
        }
        pathSegment.remainingBounces--;
        return;
    }

    // Perfect mirror
    if (m.hasReflective > 0.f) {
        glm::vec3 wr = glm::reflect(-wo, n);
        pathSegment.ray.origin = intersect + n * 1e-4f;
        pathSegment.ray.direction = glm::normalize(wr);
        pathSegment.color *= m.color;
        pathSegment.remainingBounces--;
        return;
    }

    // Diffuse (Lambert)
    glm::vec3 wi = calculateRandomDirectionInHemisphere(n, rng);
    pathSegment.ray.origin = intersect + n * 1e-4f;
    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.color *= m.color; // Cos-weighted BRDF/PDF cancel
    pathSegment.remainingBounces--;
}
