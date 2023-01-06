#pragma once

#include "LinkedHorizonCommon.h"

namespace Ecila
{

	class RaySceneIntersection
	{
    public:
        RaySceneIntersection() = default;
        RaySceneIntersection(const Point3f& p, const Normal3f& n, const Vector3f& pError,
            const Vector3f& wo, Float time,
            const MediumInterface& mediumInterface)
            : p(p),
            time(time),
            pError(pError),
            wo(Normalize(wo)),
            n(n),
            mediumInterface(mediumInterface) {}
        bool IsValid() const
        {
            return;
        }
        bool IsSurfaceInteraction() const { return n != Normal3f(); }
        Ray SpawnRay(const Vector3f& d) const {
            Point3f o = OffsetRayOrigin(p, pError, n, d);
            return Ray(o, d, Infinity, time, GetMedium(d));
        }
        Ray SpawnRayTo(const Point3f& p2) const {
            Point3f origin = OffsetRayOrigin(p, pError, n, p2 - p);
            Vector3f d = p2 - p;
            return Ray(origin, d, 1 - ShadowEpsilon, time, GetMedium(d));
        }
        Ray SpawnRayTo(const Interaction& it) const {
            Point3f origin = OffsetRayOrigin(p, pError, n, it.p - p);
            Point3f target = OffsetRayOrigin(it.p, it.pError, it.n, origin - it.p);
            Vector3f d = target - origin;
            return Ray(origin, d, 1 - ShadowEpsilon, time, GetMedium(d));
        }
        Interaction(const Point3f& p, const Vector3f& wo, Float time,
            const MediumInterface& mediumInterface)
            : p(p), time(time), wo(wo), mediumInterface(mediumInterface) {}
        Interaction(const Point3f& p, Float time,
            const MediumInterface& mediumInterface)
            : p(p), time(time), mediumInterface(mediumInterface) {}
        bool IsMediumInteraction() const { return !IsSurfaceInteraction(); }
        const Medium* GetMedium(const Vector3f& w) const {
            return Dot(w, n) > 0 ? mediumInterface.outside : mediumInterface.inside;
        }
        const Medium* GetMedium() const {
            CHECK_EQ(mediumInterface.inside, mediumInterface.outside);
            return mediumInterface.inside;
        }
    public:
        Real mTime = 0;
        Vector3 p;
        Vector3 pError;
        Vector3 mNormal;
        Vector3 mWo;
        const MediumInterface* mMediumInterface = nullptr;
        Medium mMedium;
	};
}
