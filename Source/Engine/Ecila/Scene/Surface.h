#pragma once

#include "EcilaCommon.h"
#include "EcilaMath.h"
#include "Ray.h"
#include "Intersection.h"

namespace Ecila
{
    struct HitRecord
    {
        float tHit;
        uint32 primitiveID;
    };

    class Surface 
    {
    public:

        Surface(Vector3 _v0, Vector3 _v1, Vector3 _v2, Material::SharedPtr _m = nullptr)
            : v0(_v0), v1(_v1), v2(_v2), pMaterial(_m)
        {
            e1 = v1 - v0;
            e2 = v2 - v0;
            normal = e1.cross(e2).normalized();
            area = e1.cross(e2).norm() * 0.5f;
        }

        AABB GetBounds() const
        {
            return bounds;
        }

        Real getArea() const { return area; }

        Material::SharedPtr getMaterial() { return pMaterial; }

        void sample(HitRecord& rec, Real& pdf) const
        {
            Real x = std::sqrt(random());
            Real y = random();
            rec.point = v0 * (1.0f - x) + v1 * (x * (1.0f - y)) + v2 * (x * y);
            rec.normal = normal;
            rec.pMaterial = pMaterial;
            pdf = 1.0f / area;
        }

    private:
        Vector3 v0, v1, v2; // vertices A, B ,C , counter-clockwise order
        Vector3 e1, e2;     // 2 edges v1-v0, v2-v0;
        Vector3 t0, t1, t2; // texture coords
        Vector3 normal;
        float area;
        Material* material;
        AABB bounds;
    };
}
