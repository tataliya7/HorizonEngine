#include "Hittable.h"

bool hittable_list::getAABB(AABB& outputAABB) const
{
    if (objects.empty()) return false;

    AABB temp_box;
    bool first_box = true;

    for (const auto& object : objects)
    {
        if (!object->getAABB(temp_box)) return false;
        outputAABB = first_box ? temp_box : AABB::Union(outputAABB, temp_box);
        first_box = false;
    }
    return true;
}

bool hittable_list::hit(const Ray& r, Real t_min, Real t_max, HitRecord& rec) const
{
    HitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (const auto& object : objects)
    {
        if (object->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}