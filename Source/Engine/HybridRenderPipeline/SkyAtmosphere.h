#pragma once

#include "HybridRenderPipelineCommon.h"

namespace HE
{

#ifdef HIGH_QUALITY_SKY_ATMOSPHERE
#define SKY_ATMOSPHERE_DEFAULT_LUT_FORMAT PixelFormat::RGBA32Float
#else
#define SKY_ATMOSPHERE_DEFAULT_LUT_FORMAT PixelFormat::RGBA16Float
#endif

struct RenderGraphSkyAtmosphereData
{
    RenderGraphTextureHandle transmittanceLut;
    RenderGraphTextureHandle multipleScatteringLut;
    RenderGraphTextureHandle skyViewLut;
    RenderGraphTextureHandle aerialPerspectiveVolume;
};
RENDER_GRAPH_BLACKBOARD_REGISTER_STRUCT(RenderGraphSkyAtmosphereData);

struct SkyAtmosphereConfig
{
    uint32 transmittanceLutWidth = 256;
    uint32 transmittanceLutHeight = 64;
    uint32 multipleScatteringLutSize = 32;
    uint32 skyViewLutWidth = 192;
    uint32 skyViewLutHeight = 108;
    uint32 aerialPerspectiveVolumeSize = 32;

    float rayMarchMinSPP = 4;
    float rayMarchMaxSPP = 16;
};

struct DensityProfileLayer
{
    float width;
    float expTerm;
    float expScale;
    float linearTerm;
    float constantTerm;
};

struct SkyAtmosphereComponent
{
    SkyAtmosphereComponent();

    /// Platnet

    /// Platnet top at absolute world origine.
    
    /// The distance (kilometers) between the planet center and the ground.
    //[[ attr1 ]]
    float groundRadius;

    /// The average albedo of the ground.
    //[[ attr2 ]]
    Vector3 groundAlbedo;

    /// Atmosphere
    
    /// The distance (kilometers) between the ground and the top of the atmosphere.
    float atmosphereHeight;

    /// Render multiple scattering if multipleScatteringFactor > 0
    float multipleScatteringFactor;

    /// Atmosphere - Rayleigh

    /// The scattering coefficient of air molecules at the altitude where their
    /// density is maximum (usually the bottom of the atmosphere), as a function of
    /// wavelength.
    Vector3 rayleighScattering;

    float rayleighScaleHeight;

    /// Atmosphere - Mie

    /// The scattering coefficient of aerosols at the altitude where their density
    /// is maximum (usually the bottom of the atmosphere), as a function of
    /// wavelength.
    Vector3 mieScattering;

    /// The extinction coefficient of aerosols at the altitude where their density
    /// is maximum (usually the bottom of the atmosphere), as a function of
    /// wavelength.
    Vector3 mieExtinction;

    /// The asymetry parameter for the Cornette-Shanks phase function for the aerosols.
    float mieAnisotropy;

    float mieScaleHeight;

    /// Atmosphere - Absorption

    DensityProfileLayer absorptionDensity[2];

    /// The extinction coefficient of molecules that absorb light (e.g. ozone) at
    /// the altitude where their density is maximum, as a function of wavelength.
    /// The extinction coefficient at altitude h is equal to
    /// 'absorption_extinction' times 'absorption_density' at this altitude.
    Vector3 absorptionExtinction;

    /// The cosine of the maximum Sun zenith angle for which atmospheric scattering
    /// must be precomputed (for maximum precision, use the smallest Sun zenith
    /// angle yielding negligible sky light radiance values. For instance, for the
    /// Earth case, 102 degrees is a good choice - yielding mu_s_min = -0.2).
    float cosMaxSunZenithAngle;
};

struct SkyAtmosphere
{
    SkyAtmosphereConfig config;
    RenderBackend* renderBackend;

    RenderBackendShaderHandle transmittanceLutShader;
    RenderBackendShaderHandle multipleScatteringLutShader;
    RenderBackendShaderHandle skyViewLutShader;
    RenderBackendShaderHandle aerialPerspectiveVolumeShader;
    RenderBackendShaderHandle renderSkyShader;

    RenderBackendBufferHandle skyAtmosphereConstants;
};

SkyAtmosphere* CreateSkyAtmosphere(RenderBackend* renderBackend, ShaderCompiler* compiler, SkyAtmosphereConfig* config);
void DestroySkyAtmosphere(SkyAtmosphere* skyAtmosphere);

void RenderSky(RenderGraph& renderGraph, const SkyAtmosphere& skyAtmosphere, const SkyAtmosphereComponent& component);

}