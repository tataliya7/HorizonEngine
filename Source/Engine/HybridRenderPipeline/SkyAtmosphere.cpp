#include "ECS/ECS.h"
#include "SkyAtmosphere.h"

namespace HE
{

struct SkyAtmosphereConstants
{
    float bottomRadius;
    float topRadius;
    Vector3 groundAlbedo;
    Vector3 rayleighScattering;
    Vector3 mieScattering;
    Vector3 mieExtinction;
    Vector3 mieAbsorption;
    float miePhaseFunctionG;
    Vector3 absorptionExtinction;
    float rayleighDensity[12];
    float mieDensity[12];
    float absorptionDensity[12];
    float cosMaxSunZenithAngle;
    float multipleScatteringFactor;
    float rayMarchMinSPP;
    float rayMarchMaxSPP;
    uint32 transmittanceLutWidth;
    uint32 transmittanceLutHeight;
    uint32 multipleScatteringLutSize;
    uint32 skyViewLutWidth;
    uint32 skyViewLutHeight;
    uint32 aerialPerspectiveVolumeSize;
};

static void UpdateSkyAtmosphereConstants(const SkyAtmosphere& skyAtmosphere, const SkyAtmosphereComponent& component)
{
    const SkyAtmosphereConfig& config = skyAtmosphere.config;
    const float rayleighScatteringScale = 1.0f;

    SkyAtmosphereConstants constants;
    memset(&constants, 0xBA, sizeof(SkyAtmosphereConstants));
    constants.bottomRadius = component.groundRadius;
    constants.topRadius = component.groundRadius + component.atmosphereHeight;
    constants.groundAlbedo = component.groundAlbedo;
    constants.rayleighScattering = rayleighScatteringScale * component.rayleighScattering;
    constants.mieScattering = component.mieScattering;
    constants.mieExtinction = component.mieExtinction;
    constants.mieAbsorption = component.mieExtinction - component.mieScattering;
    constants.mieAbsorption.x = Math::Max(constants.mieAbsorption.x, 0.0f);
    constants.mieAbsorption.y = Math::Max(constants.mieAbsorption.y, 0.0f);
    constants.mieAbsorption.z = Math::Max(constants.mieAbsorption.z, 0.0f);
    constants.miePhaseFunctionG = component.mieAnisotropy;
    constants.absorptionExtinction = component.absorptionExtinction;
    constants.rayleighDensity[7] = -1.0f / component.rayleighScaleHeight;
    constants.mieDensity[7] = -1.0f / component.mieScaleHeight;
    memcpy(constants.absorptionDensity, &component.absorptionDensity, sizeof(component.absorptionDensity));
    constants.cosMaxSunZenithAngle = component.cosMaxSunZenithAngle;
    constants.multipleScatteringFactor = component.multipleScatteringFactor;
    constants.transmittanceLutWidth = config.transmittanceLutWidth;
    constants.transmittanceLutHeight = config.transmittanceLutHeight;
    constants.multipleScatteringLutSize = config.multipleScatteringLutSize;
    constants.skyViewLutWidth = config.skyViewLutWidth;
    constants.skyViewLutHeight = config.skyViewLutHeight;
    constants.aerialPerspectiveVolumeSize = config.aerialPerspectiveVolumeSize;
    constants.rayMarchMinSPP = config.rayMarchMinSPP;
    constants.rayMarchMaxSPP = config.rayMarchMaxSPP;

    void* data = nullptr;
    RenderBackendMapBuffer(skyAtmosphere.renderBackend, skyAtmosphere.skyAtmosphereConstants, &data);
    memcpy((uint8*)data, &constants, sizeof(constants));
    RenderBackendUnmapBuffer(skyAtmosphere.renderBackend, skyAtmosphere.skyAtmosphereConstants);
}

static void Update(RenderGraph& renderGraph, const SkyAtmosphere& skyAtmosphere, const SkyAtmosphereComponent& component)
{
    UpdateSkyAtmosphereConstants(skyAtmosphere, component);

    auto& skyAtmosphereData = renderGraph.blackboard.CreateSingleton<RenderGraphSkyAtmosphereData>();

    const RenderBackendTextureDesc transmittanceLutDesc = RenderBackendTextureDesc::Create2D(
        skyAtmosphere.config.transmittanceLutWidth,
        skyAtmosphere.config.transmittanceLutHeight,
        SKY_ATMOSPHERE_DEFAULT_LUT_FORMAT,
        TextureCreateFlags::UnorderedAccess | TextureCreateFlags::ShaderResource);
    const RenderBackendTextureDesc multipleScatteringLutDesc = RenderBackendTextureDesc::Create2D(
        skyAtmosphere.config.multipleScatteringLutSize,
        skyAtmosphere.config.multipleScatteringLutSize,
        SKY_ATMOSPHERE_DEFAULT_LUT_FORMAT,
        TextureCreateFlags::UnorderedAccess | TextureCreateFlags::ShaderResource);
    const RenderBackendTextureDesc skyViewLutDesc = RenderBackendTextureDesc::Create2D(
        skyAtmosphere.config.skyViewLutWidth,
        skyAtmosphere.config.skyViewLutHeight,
        SKY_ATMOSPHERE_DEFAULT_LUT_FORMAT,
        TextureCreateFlags::UnorderedAccess | TextureCreateFlags::ShaderResource);
    const RenderBackendTextureDesc aerialPerspectiveVolumeDesc = RenderBackendTextureDesc::Create3D(
        skyAtmosphere.config.aerialPerspectiveVolumeSize,
        skyAtmosphere.config.aerialPerspectiveVolumeSize,
        skyAtmosphere.config.aerialPerspectiveVolumeSize,
        SKY_ATMOSPHERE_DEFAULT_LUT_FORMAT,
        TextureCreateFlags::UnorderedAccess | TextureCreateFlags::ShaderResource);

    skyAtmosphereData.transmittanceLut        = renderGraph.CreateTexture(transmittanceLutDesc, "Sky Atmosphere Transmittance Lut");
    skyAtmosphereData.multipleScatteringLut   = renderGraph.CreateTexture(multipleScatteringLutDesc, "Sky Atmosphere Multiple Scattering Lut");
    skyAtmosphereData.skyViewLut              = renderGraph.CreateTexture(skyViewLutDesc, "Sky Atmosphere Sky View Lut");
    skyAtmosphereData.aerialPerspectiveVolume = renderGraph.CreateTexture(aerialPerspectiveVolumeDesc, "Sky Atmosphere Aerial Perspective Volume");
}

static void RenderTransmittanceLut(RenderGraph& renderGraph, const SkyAtmosphere& skyAtmosphere)
{
    RenderGraphBlackboard& blackboard = renderGraph.blackboard;
    renderGraph.AddPass("Render Transmittance Lut Pass", RenderGraphPassFlags::Compute,
    [&](RenderGraphBuilder& builder)
    {
        const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
        auto& skyAtmosphereData = blackboard.Get<RenderGraphSkyAtmosphereData>();

        auto& transmittanceLut = skyAtmosphereData.transmittanceLut = builder.WriteTexture(skyAtmosphereData.transmittanceLut, RenderBackendResourceState::UnorderedAccess);

        return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
        {
            uint32 dispatchWidth = CEIL_DIV(skyAtmosphere.config.transmittanceLutWidth, 8);
            uint32 dispatchHeight = CEIL_DIV(skyAtmosphere.config.transmittanceLutHeight, 8);

            ShaderArguments shaderArguments = {}; 
            shaderArguments.BindBuffer(0, perFrameData.buffer, 0);
            shaderArguments.BindBuffer(1, skyAtmosphere.skyAtmosphereConstants, 0);
            shaderArguments.BindTextureUAV(2, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(transmittanceLut)));

            commandList.Dispatch2D(
                skyAtmosphere.transmittanceLutShader,
                shaderArguments,
                dispatchWidth,
                dispatchHeight);
        };
    });
}

static void RenderMultipleScatteringLut(RenderGraph& renderGraph, const SkyAtmosphere& skyAtmosphere)
{
    RenderGraphBlackboard& blackboard = renderGraph.blackboard;
    renderGraph.AddPass("Render Multilpe Scattering Lut Pass", RenderGraphPassFlags::Compute,
    [&](RenderGraphBuilder& builder)
    { 
        const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
        auto& skyAtmosphereData = blackboard.Get<RenderGraphSkyAtmosphereData>();

        auto transmittanceLut = builder.ReadTexture(skyAtmosphereData.transmittanceLut, RenderBackendResourceState::ShaderResource);
        auto multipleScatteringLut = skyAtmosphereData.multipleScatteringLut = builder.WriteTexture(skyAtmosphereData.multipleScatteringLut, RenderBackendResourceState::UnorderedAccess);

        return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
        {
            uint32 dispatchX = skyAtmosphere.config.multipleScatteringLutSize;
            uint32 dispatchY = skyAtmosphere.config.multipleScatteringLutSize;
            uint32 dispatchZ = 1;

            ShaderArguments shaderArguments = {};
            shaderArguments.BindBuffer(0, perFrameData.buffer, 0);
            shaderArguments.BindBuffer(1, skyAtmosphere.skyAtmosphereConstants, 0);
            shaderArguments.BindTextureSRV(3, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(transmittanceLut)));
            shaderArguments.BindTextureUAV(4, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(multipleScatteringLut)));

            commandList.Dispatch(
                skyAtmosphere.multipleScatteringLutShader,
                shaderArguments,
                dispatchX,
                dispatchY,
                dispatchZ);
        };
    });
}

static void RenderSkyViewLut(RenderGraph& renderGraph, const SkyAtmosphere& skyAtmosphere)
{
    RenderGraphBlackboard& blackboard = renderGraph.blackboard;
    renderGraph.AddPass("Render Sky View Lut Pass", RenderGraphPassFlags::Compute,
    [&](RenderGraphBuilder& builder)
    {
        const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
        auto& skyAtmosphereData = blackboard.Get<RenderGraphSkyAtmosphereData>();

        auto transmittanceLut = builder.ReadTexture(skyAtmosphereData.transmittanceLut, RenderBackendResourceState::ShaderResource);
        auto multipleScatteringLut = builder.ReadTexture(skyAtmosphereData.multipleScatteringLut, RenderBackendResourceState::ShaderResource);

        auto skyViewLut = skyAtmosphereData.skyViewLut = builder.WriteTexture(skyAtmosphereData.skyViewLut, RenderBackendResourceState::UnorderedAccess);

        return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
        {
            uint32 dispatchWidth = CEIL_DIV(skyAtmosphere.config.skyViewLutWidth, 8);
            uint32 dispatchHeight = CEIL_DIV(skyAtmosphere.config.skyViewLutHeight, 8);

            ShaderArguments shaderArguments = {};
            shaderArguments.BindBuffer(0, perFrameData.buffer, 0);
            shaderArguments.BindBuffer(1, skyAtmosphere.skyAtmosphereConstants, 0);
            shaderArguments.BindTextureSRV(3, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(transmittanceLut)));
            shaderArguments.BindTextureSRV(5, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(multipleScatteringLut)));
            shaderArguments.BindTextureUAV(6, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(skyViewLut)));

            commandList.Dispatch2D(
                skyAtmosphere.skyViewLutShader,
                shaderArguments,
                dispatchWidth,
                dispatchHeight);
        };
    });
}

static void RenderAerialPerspectiveVolume(RenderGraph& renderGraph, const SkyAtmosphere& skyAtmosphere)
{
    RenderGraphBlackboard& blackboard = renderGraph.blackboard;
    renderGraph.AddPass("Render Aerial Perspective Volume Pass", RenderGraphPassFlags::AsyncCompute,
    [&](RenderGraphBuilder& builder)
    { 
        const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
        auto& skyAtmosphereData = blackboard.Get<RenderGraphSkyAtmosphereData>();

        auto transmittanceLut = builder.ReadTexture(skyAtmosphereData.transmittanceLut, RenderBackendResourceState::ShaderResource);
        auto multipleScatteringLut = builder.ReadTexture(skyAtmosphereData.multipleScatteringLut, RenderBackendResourceState::ShaderResource);
        auto& aerialPerspectiveVolume = skyAtmosphereData.aerialPerspectiveVolume = builder.WriteTexture(skyAtmosphereData.aerialPerspectiveVolume, RenderBackendResourceState::UnorderedAccess);

        return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
        {
            uint32 dispatchX = skyAtmosphere.config.aerialPerspectiveVolumeSize / 8;
            uint32 dispatchY = skyAtmosphere.config.aerialPerspectiveVolumeSize / 8;
            uint32 dispatchZ = skyAtmosphere.config.aerialPerspectiveVolumeSize / 4;

            ShaderArguments shaderArguments = {};
            shaderArguments.BindBuffer(0, perFrameData.buffer, 0);
            shaderArguments.BindBuffer(1, skyAtmosphere.skyAtmosphereConstants, 0);
            shaderArguments.BindTextureSRV(3, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(transmittanceLut)));
            shaderArguments.BindTextureSRV(5, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(multipleScatteringLut)));
            shaderArguments.BindTextureUAV(8, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(aerialPerspectiveVolume)));

            commandList.Dispatch(
                skyAtmosphere.aerialPerspectiveVolumeShader,
                shaderArguments,
                dispatchX,
                dispatchY,
                dispatchZ);
        };
    });
}

static void RenderSkyRayMarching(RenderGraph& renderGraph, const SkyAtmosphere& skyAtmosphere)
{
    RenderGraphBlackboard& blackboard = renderGraph.blackboard;
    renderGraph.AddPass("Render Sky Ray Marching Pass", RenderGraphPassFlags::Compute,
    [&](RenderGraphBuilder& builder)
    { 
        const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
        auto& skyAtmosphereData = blackboard.Get<RenderGraphSkyAtmosphereData>();
        const auto& viewData = blackboard.Get<RenderGraphPerFrameData>();
        const auto& depthBufferData = blackboard.Get<RenderGraphDepthBuffer>();
        auto& sceneColorData = blackboard.Get<RenderGraphSceneColor>();

        auto depthBuffer = builder.ReadTexture(depthBufferData.depthBuffer, RenderBackendResourceState::ShaderResource);
        auto skyViewLut = builder.ReadTexture(skyAtmosphereData.skyViewLut, RenderBackendResourceState::ShaderResource);
        auto aerialPerspectiveVolume = builder.ReadTexture(skyAtmosphereData.aerialPerspectiveVolume, RenderBackendResourceState::ShaderResource);
        auto sceneColor = sceneColorData.sceneColor = builder.WriteTexture(sceneColorData.sceneColor, RenderBackendResourceState::UnorderedAccess);

        return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
        {
            uint32 dispatchWidth = CEIL_DIV(perFrameData.data.targetResolutionWidth, 8);
            uint32 dispatchHeight = CEIL_DIV(perFrameData.data.targetResolutionHeight, 8);

            ShaderArguments shaderArguments = {};
            shaderArguments.BindBuffer(0, perFrameData.buffer, 0);
            shaderArguments.BindBuffer(1, skyAtmosphere.skyAtmosphereConstants, 0);
            shaderArguments.BindTextureSRV(11, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(depthBuffer)));
            shaderArguments.BindTextureSRV(7, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(skyViewLut)));
            shaderArguments.BindTextureSRV(9, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(aerialPerspectiveVolume)));
            shaderArguments.BindTextureUAV(10, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(sceneColor)));

            commandList.Dispatch2D(
                skyAtmosphere.renderSkyShader,
                shaderArguments,
                dispatchWidth,
                dispatchHeight);
        };
    });
}

void RenderSky(RenderGraph& renderGraph, const SkyAtmosphere& skyAtmosphere, const SkyAtmosphereComponent& component)
{
    Update(renderGraph, skyAtmosphere, component);
    RenderTransmittanceLut(renderGraph, skyAtmosphere);
    RenderMultipleScatteringLut(renderGraph, skyAtmosphere);
    RenderSkyViewLut(renderGraph, skyAtmosphere);
    RenderAerialPerspectiveVolume(renderGraph, skyAtmosphere);
    RenderSkyRayMarching(renderGraph, skyAtmosphere);
}

SkyAtmosphere* CreateSkyAtmosphere(RenderBackend* renderBackend, ShaderCompiler* compiler, SkyAtmosphereConfig* config)
{
    SkyAtmosphere* skyAtmosphere = new SkyAtmosphere();
    skyAtmosphere->renderBackend = renderBackend;
    skyAtmosphere->config = *config;

    uint32 deviceMask = ~0u;

    RenderBackendBufferDesc skyAtmosphereConstantBufferDesc = RenderBackendBufferDesc::CreateByteAddress(sizeof(SkyAtmosphereConstants));
    skyAtmosphere->skyAtmosphereConstants = RenderBackendCreateBuffer(renderBackend, deviceMask, &skyAtmosphereConstantBufferDesc, "SkyAtmosphereConstants");

    std::vector<uint8> source;
    std::vector<const wchar*> includeDirs;
    std::vector<const wchar*> defines;
    includeDirs.push_back(HE_TEXT("../../../Shaders/HybridRenderPipeline"));

    LoadShaderSourceFromFile("../../../Shaders/HybridRenderPipeline/SkyAtmosphereTransmittanceLut.hsf", source);
    RenderBackendShaderDesc transmittanceLutShaderDesc;
    CompileShader(
        compiler,
        source,
        HE_TEXT("SkyAtmosphereTransmittanceLutCS"),
        RenderBackendShaderStage::Compute,
        ShaderRepresentation::SPIRV,
        includeDirs,
        defines,
        &transmittanceLutShaderDesc.stages[(uint32)RenderBackendShaderStage::Compute]);
    transmittanceLutShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Compute] = "SkyAtmosphereTransmittanceLutCS";
    skyAtmosphere->transmittanceLutShader = RenderBackendCreateShader(renderBackend, deviceMask, &transmittanceLutShaderDesc, "SkyAtmosphereTransmittanceLutCS");

    LoadShaderSourceFromFile("../../../Shaders/HybridRenderPipeline/SkyAtmosphereMultipleScatteringLut.hsf", source);
    RenderBackendShaderDesc multipleScatteringLutShaderDesc;
    CompileShader(
        compiler,
        source,
        HE_TEXT("SkyAtmosphereMultipleScatteringLutCS"),
        RenderBackendShaderStage::Compute,
        ShaderRepresentation::SPIRV,
        includeDirs,
        defines,
        &multipleScatteringLutShaderDesc.stages[(uint32)RenderBackendShaderStage::Compute]);
    multipleScatteringLutShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Compute] = "SkyAtmosphereMultipleScatteringLutCS";
    skyAtmosphere->multipleScatteringLutShader = RenderBackendCreateShader(renderBackend, deviceMask, &multipleScatteringLutShaderDesc, "SkyAtmosphereMultipleScatteringLutCS");

    LoadShaderSourceFromFile("../../../Shaders/HybridRenderPipeline/SkyAtmosphereSkyViewLut.hsf", source);
    RenderBackendShaderDesc skyViewLutShaderDesc;
    CompileShader(
        compiler,
        source,
        HE_TEXT("SkyAtmosphereSkyViewLutCS"),
        RenderBackendShaderStage::Compute,
        ShaderRepresentation::SPIRV,
        includeDirs,
        defines,
        &skyViewLutShaderDesc.stages[(uint32)RenderBackendShaderStage::Compute]);
    skyViewLutShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Compute] = "SkyAtmosphereSkyViewLutCS";
    skyAtmosphere->skyViewLutShader = RenderBackendCreateShader(renderBackend, deviceMask, &skyViewLutShaderDesc, "SkyAtmosphereSkyViewLutCS");

    LoadShaderSourceFromFile("../../../Shaders/HybridRenderPipeline/SkyAtmosphereAerialPerspectiveVolume.hsf", source);
    RenderBackendShaderDesc aerialPerspectiveVolumeShaderDesc;
    CompileShader(
        compiler,
        source,
        HE_TEXT("SkyAtmosphereAerialPerspectiveVolumeCS"),
        RenderBackendShaderStage::Compute,
        ShaderRepresentation::SPIRV,
        includeDirs,
        defines,
        &aerialPerspectiveVolumeShaderDesc.stages[(uint32)RenderBackendShaderStage::Compute]);
    aerialPerspectiveVolumeShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Compute] = "SkyAtmosphereAerialPerspectiveVolumeCS";
    skyAtmosphere->aerialPerspectiveVolumeShader = RenderBackendCreateShader(renderBackend, deviceMask, &aerialPerspectiveVolumeShaderDesc, "SkyAtmosphereAerialPerspectiveVolumeCS");

    LoadShaderSourceFromFile("../../../Shaders/HybridRenderPipeline/SkyAtmosphereRenderSky.hsf", source);
    RenderBackendShaderDesc renderSkyShaderDesc;
    CompileShader(
        compiler,
        source,
        HE_TEXT("SkyAtmosphereRenderSkyCS"),
        RenderBackendShaderStage::Compute,
        ShaderRepresentation::SPIRV,
        includeDirs,
        defines,
        &renderSkyShaderDesc.stages[(uint32)RenderBackendShaderStage::Compute]);
    renderSkyShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Compute] = "SkyAtmosphereRenderSkyCS";
    skyAtmosphere->renderSkyShader = RenderBackendCreateShader(renderBackend, deviceMask, &renderSkyShaderDesc, "SkyAtmosphereRenderSkyCS");

    //skyAtmosphere->renderTransmittanceLutShader        = skyAtmosphere->shaderSystem->LoadShader("SkyAtmosphereTransmittanceLut.hsf",        "SkyAtmosphereTransmittanceLutCS",        nullptr);
    //skyAtmosphere->renderMultipleScatteringLutShader   = skyAtmosphere->shaderSystem->LoadShader("SkyAtmosphereMultipleScatteringLut.hsf",   "SkyAtmosphereSkyViewLutCS",              nullptr);
    //skyAtmosphere->renderSkyViewLutShader              = skyAtmosphere->shaderSystem->LoadShader("SkyAtmosphereSkyViewLut.hsf",              "SkyAtmosphereMultipleScatteringLutCS",   nullptr);
    //skyAtmosphere->renderAerialPerspectiveVolumeShader = skyAtmosphere->shaderSystem->LoadShader("SkyAtmosphereAerialPerspectiveVolume.hsf", "SkyAtmosphereAerialPerspectiveVolumeCS", nullptr);
    //skyAtmosphere->renderSkyShader                     = skyAtmosphere->shaderSystem->LoadShader("SkyAtmosphereRenderSky.hsf",               "SkyAtmosphereRenderSkyCS",               nullptr);
    
    return skyAtmosphere;
}

void DestroySkyAtmosphere(SkyAtmosphere* skyAtmosphere)
{
    RenderBackend* renderBackend = skyAtmosphere->renderBackend;
    RenderBackendDestroyBuffer(renderBackend, skyAtmosphere->skyAtmosphereConstants);
    RenderBackendDestroyShader(renderBackend, skyAtmosphere->transmittanceLutShader);
    RenderBackendDestroyShader(renderBackend, skyAtmosphere->multipleScatteringLutShader);
    RenderBackendDestroyShader(renderBackend, skyAtmosphere->skyViewLutShader);
    RenderBackendDestroyShader(renderBackend, skyAtmosphere->aerialPerspectiveVolumeShader);
    RenderBackendDestroyShader(renderBackend, skyAtmosphere->renderSkyShader);
    delete skyAtmosphere;
}

SkyAtmosphereComponent::SkyAtmosphereComponent()
{
    using namespace entt;
    auto factory = entt::meta<SkyAtmosphereComponent>();//.type("Sky Atmosphere"_hs)
    factory.data<&SkyAtmosphereComponent::groundRadius, entt::as_ref_t>("Ground Radius"_hs)
           .prop("Name"_hs, std::string("Ground Radius"));
    factory.data<&SkyAtmosphereComponent::groundAlbedo, entt::as_ref_t>("Ground Albedo"_hs)
           .prop("Name"_hs, std::string("Ground Albedo"));
    factory.data<&SkyAtmosphereComponent::atmosphereHeight, entt::as_ref_t>("Atmosphere Height"_hs)
        .prop("Name"_hs, std::string("Atmosphere Height"));
    factory.data<&SkyAtmosphereComponent::multipleScatteringFactor, entt::as_ref_t>("Multiple Scattering Factor"_hs)
        .prop("Name"_hs, std::string("Multiple Scattering Factor"));
    factory.data<&SkyAtmosphereComponent::rayleighScattering, entt::as_ref_t>("Rayleigh Scattering"_hs)
        .prop("Name"_hs, std::string("Rayleigh Scattering"));
}

}