#pragma once

#include "EcilaCommon.h"
#include "EcilaMath.h"
#include "Ray.h"

namespace Ecila
{
    struct AABB
    {
        Vector3 minPoint;
        Vector3 maxPoint;
        AABB() : minPoint(0.0f), maxPoint(0.0f) {}
        AABB(const Vector3& min, const Vector3& max) : minPoint(min), maxPoint(max) {}
        static AABB CreateFromTwoPoints(const Vector3 p1, const Vector3 p2)
        {
            return AABB(min3(p1, p2), max3(p1, p2));
        }
        static AABB Union(const AABB& bb1, const AABB& bb2)
        {
            return AABB(min3(bb1.minPoint, bb2.minPoint), max3(bb1.maxPoint, bb2.maxPoint));
        }
        static AABB Union(const AABB& bb, const Vector3& p)
        {
            return AABB(min3(bb.minPoint, p), max3(bb.maxPoint, p));
        }
    };

}