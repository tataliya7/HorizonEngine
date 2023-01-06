#pragma once

#include "LinkedHorizonCommon.h"

namespace Ecila
{
	class Sphere
	{
	public:

		static SharedPtr<Sphere> Create();

		~Sphere();

	private:

		Sphere();
	};
}
