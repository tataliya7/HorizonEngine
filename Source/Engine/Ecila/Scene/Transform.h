#pragma once

#include "HorizonCommon.h"
#include "HorizonMath.h"

namespace Horizon
{
	struct Transform
	{
		Matrix4 matrix = Matrix4(1);
		Matrix4 inverseTransposeMatrix = Matrix4(1);
	};
}
