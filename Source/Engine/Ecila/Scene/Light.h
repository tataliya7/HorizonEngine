#pragma once

#include "EcilaCommon.h"
#include "EcilaMath.h"
#include "Medium.h"

namespace Ecila
{
	enum class LightType
	{
		Directional = 0,
		RectangularArea = 1,
	};

	struct Light
	{
		LightType type;
		Vector3 color;
		float intensity;
		Vector3 direction;
		Vector3 rectangle[4];
	};

}
