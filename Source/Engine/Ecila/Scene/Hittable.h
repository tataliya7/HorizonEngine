#pragma once

#include "Ray.h"
#include "AABB.h"

class Material;
using MaterialPtr = std::shared_ptr<Material>;

struct IntersectionInfo
{

};

struct HitRecord
{
    Vector3 point;
    Real t;
    MaterialPtr pMaterial;
    Vector3 normal;
    bool frontFace;
    void setFaceNormal(const Ray& ray, const Vector3& outwardNormal)
    {
        frontFace = ray.getDirection().dot(outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

class Hittable
{
public:
    using SharedPtr = std::shared_ptr<Hittable>;
    virtual bool hit(const Ray& ray, Real tMin, Real tMax, HitRecord& record) const = 0;
    virtual bool getAABB(AABB& outputAABB) const = 0;
};

class hittable_list : public Hittable
{
public:
    hittable_list() {}
    hittable_list(const std::vector<std::shared_ptr<Hittable>>& objects) :objects(objects) { }
    
    void clear() { objects.clear(); }
    void add(std::shared_ptr<Hittable> object) { objects.push_back(object); }

    virtual bool hit(const Ray& r, Real t_min, Real t_max, HitRecord& rec) const override;
    virtual bool getAABB(AABB& outputAABB) const override;
public:
    std::vector<std::shared_ptr<Hittable>> objects;
};
