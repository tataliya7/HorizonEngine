#include "Sampler.h"

namespace Ecila
{
	SharedPtr<Sampler> Sampler::Create1D(SequenceGeneratorType sgType)
	{
		return SharedPtr<Sampler>(new Sampler(SamplerType::Sampler1D, sgType));
	}

	SharedPtr<Sampler> Sampler::Create2D(SequenceGeneratorType sgType)
	{
		return SharedPtr<Sampler>(new Sampler(SamplerType::Sampler2D, sgType));
	}

	Sampler::Sampler(SamplerType type, SequenceGeneratorType sgType)
		: mType(type)
	{

	}

	void Sampler::StartPixelSample2D(const Vector2i& pos, int index)
	{
		mCurrentSampleCount = 0;
	}
}
