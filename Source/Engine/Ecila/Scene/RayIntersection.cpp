#include "RayIntersection.h"

namespace Ecila
{
	float RayAabbIntersection(Vector3 rayOrigin, Vector3 rayDirection, const AABB& aabb)
	{
		float t = std::numeric_limits<float>::max();
		Vector3 invDir = 1.0f / rayDirection;
		Vector3 tBot = (aabb.minPoint - rayOrigin) * invDir;
		Vector3 tTop = (aabb.maxPoint - rayOrigin) * invDir;
		float t0 = glm::compMax(glm::min(tBot, tTop));
		float t1 = glm::compMin(glm::max(tBot, tTop));
		return t1 > std::max(t0, 0.0f) ? t0 : -1.0f;
	}

	float RayTriangleIntersection(Vector3 rayOrigin, Vector3 rayDirection, Vector3 v0, Vector3 v1, Vector3 v2)
	{
		Vector3 e1 = v1 - v0;
		Vector3 e2 = v2 - v0;

		Vector3 p = Math::Cross(rayDirection, e2);
		float det = Math::Dot(e1, p);
		float invDet = 1.0f / det;

		if (fabs(det) < EPSILON)
		{
			return -1.0f;
		}

		Vector3 t = rayOrigin - v0;
		if (det < 0.0f)
		{
			t = -t;
			det = -det;
		}

		float u = Math::Dot(t, p);
		if (u < 0.0f || u > det)
		{
			return -1.0f;
		}

		Vector3 q = Math::Cross(t, e1);
		float v = Math::Dot(rayDirection, q);
		if (v < 0.0f || u + v > det)
		{
			return -1.0f;
		}

		return Math::Dot(e2, q) * invDet;
	}
}