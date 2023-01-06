#include "Sphere.h"

Sphere::Sphere(const Vector3& center, const Real& radius, Material::SharedPtr pMaterial)
    : mCenter(center)
    , mRadius(radius)
    , mpMaterial(pMaterial)
{
}

bool Sphere::hit(const Ray& ray, Real tMin, Real tMax, HitRecord& record) const
{
    Vector3 L = ray.getOrigin() - mCenter;
    auto a = ray.getDirection().squaredNorm();
    auto half_b = ray.getDirection().dot(L);
    auto c = L.squaredNorm() - mRadius * mRadius;
    auto discriminant = half_b * half_b - a * c;
    if (discriminant > 0) 
    {
        auto root = sqrt(discriminant);
        auto temp = (-half_b - root) / a;
        if (temp < tMax && temp > tMin)
        {
            record.t = temp;
            record.point = ray.at(record.t);
            Vector3 outwardNormal = (record.point - mCenter) / mRadius;
            record.setFaceNormal(ray, outwardNormal);
            record.pMaterial = mpMaterial;
            return true;
        }
        temp = (-half_b + root) / a;
        if (temp < tMax && temp > tMin)
        {
            record.t = temp;
            record.point = ray.at(record.t);
            Vector3 outwardNormal = (record.point - mCenter) / mRadius;
            record.setFaceNormal(ray, outwardNormal);
            record.pMaterial = mpMaterial;
            return true;
        }
    }
    return false;
}

bool Sphere::getAABB(AABB& outputAABB) const
{
    outputAABB = AABB(mCenter - Vector3(mRadius, mRadius, mRadius),
                      mCenter + Vector3(mRadius, mRadius, mRadius));
    return true;
}
