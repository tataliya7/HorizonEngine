#pragma once

#include "LinkedHorizonCommon.h"

namespace Ecila
{
	enum class SequenceGeneratorType
	{
		Halton,
		Sobol,
		MLT,
	};

	class SequenceGeneratorBase
	{
	public:
		virtual ~SequenceGeneratorBase();
	};

	class HaltonSequenceGenerator : public SequenceGeneratorBase
	{
	public:

		HaltonSequenceGenerator();

		~HaltonSequenceGenerator();

	private:

	};
}
