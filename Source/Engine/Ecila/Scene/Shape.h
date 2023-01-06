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

class Rectangle : public Hittable
{
public:
    Rectangle() = default;

    Rectangle(Real _x0, Real _x1, Real _y0, Real _y1, Real _k, Material::SharedPtr mat)
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mpMaterial(mat) {};

    bool hit(const Ray& ray, Real tMin, Real tMax, HitRecord& record) const override;
    bool getAABB(AABB& outputAABB) const override;

public:
    Real x0, x1, y0, y1, k;
    Material::SharedPtr mpMaterial;
};