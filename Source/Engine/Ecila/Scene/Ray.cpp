#include "Ray.h"

Ray::Ray(const Vector3& origin, const Vector3& dir)
    : mOrigin(origin), mDir(dir.normalized())
{

}

Vector3 Ray::at(Real t) const
{
    return mOrigin + t * mDir;
}
