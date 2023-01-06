#pragma once

#include "Hittable.h"
#include "Material.h"

class Sphere : public Hittable
{
public:

    using SharedPtr = std::shared_ptr<Sphere>;

    Sphere() = default;
    Sphere(const Vector3& center, const Real& radius, Material::SharedPtr pMaterial);

    Material::SharedPtr getMaterial() { return mpMaterial; }
    bool hit(const Ray& ray, Real tMin, Real tMax, HitRecord& record) const override;
    bool getAABB(AABB& outputAABB) const override;

protected:
    Vector3 mCenter;
    Real mRadius;
    Material::SharedPtr mpMaterial;
};
