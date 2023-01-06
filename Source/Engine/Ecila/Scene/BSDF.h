#pragma once

#include "HorizonCommon.h"

namespace Ecila
{
    struct BSDFSample 
    {
        // BSDFSample Public Methods
        BSDFSample() = default;
        BSDFSample(SampledSpectrum f, Vector3f wi, Float pdf, BxDFFlags flags, Float eta = 1,
            bool pdfIsProportional = false)
        : f(f),
        wi(wi),
        pdf(pdf),
        flags(flags),
        eta(eta),
        pdfIsProportional(pdfIsProportional) {}

        bool IsReflection() const { return pbrt::IsReflective(flags); }
        bool IsTransmission() const { return pbrt::IsTransmissive(flags); }
        bool IsDiffuse() const { return pbrt::IsDiffuse(flags); }
        bool IsGlossy() const { return pbrt::IsGlossy(flags); }
        bool IsSpecular() const { return pbrt::IsSpecular(flags); }

        std::string ToString() const;
        SampledSpectrum f;
        Vector3 wi;
        float pdf = 0;
        float eta = 1;
        bool pdfIsProportional = false;
    };

	class BSDF
	{
	public:
		BSDF() = default;
		virtual ~BSDF() = default;

        std::optional<BSDFSample> Sample(
            Vector3 wo, float u, Vector2 u2,
            TransportMode mode = TransportMode::Radiance,
            BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
            Vector3 wo = RenderToLocal(woRender);
            sample->wi = LocalToRender(bs->wi);
            if (wo.z == 0 || !(bxdf.Flags() & sampleFlags))
                return {};
            // Sample _bxdf_ and return _BSDFSample_
            std::optional<BSDFSample> sample = bxdf.Sample_f(wo, u, u2, mode, sampleFlags);
            if (sample)
            {
                DCHECK_GE(bs->pdf, 0);
            }
            if (!sample || !bs->f || bs->pdf == 0 || bs->wi.z == 0)
            {
                return {};
            }
            return sample;
        }

	private:

	};
}
