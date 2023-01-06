#pragma once

#include "EcilaCommon.h"
#include "EcilaMath.h"
#include "Bounds.h"

namespace Ecila
{

	extern float RayAabbIntersection(Vector3 rayOrigin, Vector3 rayDirection, const AABB& aabb);

	extern float RayTriangleIntersection(Vector3 rayOrigin, Vector3 rayDirection, Vector3 v0, Vector3 v1, Vector3 v2);

}