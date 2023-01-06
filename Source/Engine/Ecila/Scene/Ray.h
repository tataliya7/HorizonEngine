#pragma once

#include "EcilaCommon.h"
#include "EcilaMath.h"

namespace Ecila
{
    class Ray
    {
    public:
        Ray() : origin(0, 0, 0), direction(0, 0, 1) {}
        Ray(const Vector3& o, const Vector3& d) : origin(o), direction(d) {}
        Vector3 GetOrigin() const
        {
            return origin;
        }
        Vector3 GetDirection() const
        {
            return direction;
        }
        Vector3 operator()(float t) const 
        { 
            return origin + t * direction;
        }
        Vector3 At(Real t) const 
        { 
            return origin + t * direction; 
        }
    private:
        Vector3 origin;
        Vector3 direction;
    };

    class RayDifferential : public Ray
    {
    };
}
