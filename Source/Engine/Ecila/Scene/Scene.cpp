#include "RenderScene.h"
#include "Mesh.h"
#include "Light.h"
#include "Ray.h"
#include "BVH.h"

/*
Vector3 generate_random_unit_sphere()
{
    while (true)
    {
        auto p = randomVector3(-1, 1);
        if (p.squaredNorm() >= 1)
            continue;
        return p;
    }
}

Vector3 generate_random_unit_hemisphere(const Vector3& normal)
{
    Vector3 in_unit_sphere = generate_random_unit_sphere();
    if (in_unit_sphere.dot(normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

Vector3 generate_random_unit_vector()
{
    auto a = random(0, 2 * PI);
    auto z = random(-1, 1);
    auto r = sqrt(1 - z * z);
    return Vector3(r * cos(a), r * sin(a), z);
}

Real schlick(Real cosine, Real ref_idx)
{
    Real r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

inline Vector3 reflect(const Vector3& v, const Vector3& n)
{
    return v - 2 * v.dot(n) * n;
}

inline Vector3 refract(const Vector3& I, const Vector3& N, Real etai_over_etat)
{
    Real cosi = clamp(-I.dot(N), -1, 1);
    Vector3 r_out_parallel = etai_over_etat * (I + cosi * N);
    Vector3 r_out_perp = -sqrt(1.0f - r_out_parallel.squaredNorm()) * N;
    return r_out_parallel + r_out_perp;
}

inline void fresnel(const Vector3& I, const Vector3& N, const Real& ior, Real& kr)
{
    Real cosi = clamp(I.dot(N), -1, 1);
    Real etai = 1, etat = ior;
    if (cosi > 0) { std::swap(etai, etat); }
    // Compute sini using Snell's law
    Real sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1) {
        kr = 1;
    }
    else {
        Real cost = sqrtf(std::max(0.f, 1 - sint * sint));
        cosi = fabsf(cosi);
        Real Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        Real Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        kr = (Rs * Rs + Rp * Rp) / 2;
    }
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;
}

void Scene::build()
{
    std::vector<Hittable::SharedPtr> pHittables;
    for (auto mesh : mMeshes)
    {
        pHittables.push_back((Hittable::SharedPtr)mesh);
    }
    mpRoot = (BVHNode::SharedPtr)new BVHNode(pHittables);
}

void Scene::sampleLight(HitRecord& rec, Real& pdf) const
{
    Real emit_area_sum = 0;
    for (uint32 i = 0; i < mLights.size(); ++i)
    {
        emit_area_sum += mLights[i].getArea();
    }
    Real p = random() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32 i = 0; i < mLights.size(); ++i)
    {
        emit_area_sum += mLights[i].getArea();
        if (p <= emit_area_sum)
        {
            mLights[i].sample(rec, pdf);
            break;
        }
    }
}

// Implementation of Path Tracing
Vector3 Scene::castRay(const Ray& ray, int depth) const
{
    Vector3 hitcolor = Vector3(0, 0, 0);
    HitRecord rec;
    if (!mpRoot->hit(ray, 0.001f, std::numeric_limits<Real>::infinity(), rec))
    {
        return hitcolor;
    }

    // Deal with self-luminous material
    if (rec.pMaterial->isLuminous())
        return Vector3(1, 1, 1);

    Vector3 wo = -ray.getDirection().normalized();
    Vector3 p = rec.point;
    Vector3 N = rec.normal.normalized();

    Real pdf_light = 0.0f;
    HitRecord lightSamplePoint;
    sampleLight(lightSamplePoint, pdf_light);
    Vector3 x = lightSamplePoint.point;
    Vector3 ws = (x - p).normalized();
    Vector3 NN = lightSamplePoint.normal.normalized();

    Vector3 L_dir = Vector3(0, 0, 0);
    HitRecord test;
    // Direct light
    mpRoot->hit(Ray(p, ws), 0.001f, std::numeric_limits<Real>::infinity(), test);
    Real temp_dist = (test.point - x).norm();
    if (temp_dist < 0.01f)
    {
        Vector3 emit = lightSamplePoint.pMaterial->getEmission();
        Vector3 f_r = rec.pMaterial->BRDF(ws, wo, N);
        Real cosTheta1 = std::max(0.0f, ws.dot(N));
        Real cosTheta2 = std::max(0.0f, -ws.dot(NN));
        Real r2 = (x - p).dot(x - p);
        L_dir = emit.cwiseProduct(f_r) * cosTheta1 * cosTheta2 / r2 / pdf_light;
    }

    Vector3 L_indir = Vector3(0, 0, 0);
    Real P_RR = random();
    Real RussianRoulette = 0.8f;
    // Indirect light
    if (P_RR < RussianRoulette)
    {
        Vector3 wi = rec.pMaterial->sample(wo, N);
        Real pdf = rec.pMaterial->pdf(wi, wo, N);
        if (pdf > EPSILON)
        {
            Vector3 f_r = rec.pMaterial->BRDF(wi, wo, N);
            Real cosTheta = std::max(0.0f, wi.dot(N));
            L_indir = castRay(Ray(p, wi), depth + 1).cwiseProduct(f_r) * cosTheta / pdf;
            L_indir = L_indir / RussianRoulette;
        }
    }
    hitcolor = L_indir + L_dir;
    return hitcolor;
}
*/
namespace Ecila
{
    void Scene::BuildBVH()
    {
        bvh->Build(surfaces);
    }

    HitRecord Scene::TraversalAccelerations(const Ray& ray, float tMin, float tMax) const
    {
        std::queue<BVHNode*> queue;
        HitRecord closestHit;
        closestHit.t = std::numeric_limits<float>::max();
        closestHit.SetValid(false);

        if (!bvh->IsEmpty())
        {
            queue.push(bvh->GetRoot());
        }

        while (!queue.empty())
        {
            BVHNode* node = queue.front();
            queue.pop();

            float t = RayAabbIntersection(node->GetBounds(), ray, tMin, tMax);
            if (t < 0.0f || t > closestHit.t)
            {
                continue;
            }

            if (node->IsBottomLevelAS())
            {
                uint32 firstSurface = node->GetFirstSurface();
                uint32 numSurfaces = node->GetSurfaceCount();
                for (uint32 i = 0; i < numSurfaces; i++)
                {
                    const Surface* surface = &surfaces[firstSurface + i];
                    HitRecord hitRecord;
                    if (surface->Intersect(ray, tMin, tMax, hitRecord))
                    {
                        if (hitRecord.t < closestHit.t)
                        {
                            closestHit = hitRecord;
                        }
                    }
                }
            }
            else
            {
                if (node->leftNode)
                {
                    queue.push(node->leftNode);
                }
                if (node->rightNode)
                {
                    queue.push(node->rightNode);
                }
            }
        }
        return closestHit;
    }

    void Scene::Update()
    {
        for (const auto& mesh : meshes)
        {

        }

        for (const auto& light : lights)
        {

        }
    }
}
