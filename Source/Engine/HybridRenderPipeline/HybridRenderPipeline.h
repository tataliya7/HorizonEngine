#pragma once

#include "HybridRenderPipelineCommon.h"
#include "SkyAtmosphere.h"

#define HE_DEFUALT_RENDER_PIPELINE_NAME "Horizon Engine Hybrid Render Pipeline"

namespace HE
{

struct HybridRenderPipelineSettings
{
	float renderScale;
	Vector2u shadowMapResolution;
};

class HybridRenderPipeline : public RenderPipeline
{
public:
	HybridRenderPipeline(RenderContext* context);
	void Init();
	void SetupRenderGraph(SceneView* view, RenderGraph* renderGraph) override;
	HybridRenderPipelineSettings settings;
private:

	void RenderRayTracingShadows(
		RenderGraph& renderGraph,
		RenderGraphBlackboard& blackboard,
		const SceneView& view,
		const LightRenderProxy& lightProxy,
		RenderBackendBufferHandle parametersBuffer,
		RenderGraphTextureHandle& shadowMask);

	void RenderScreenSpaceShadows(
		RenderGraph& renderGraph,
		RenderGraphBlackboard& blackboard,
		const SceneView& view,
		const LightRenderProxy& lightProxy,
		RenderBackendBufferHandle parametersBuffer,
		RenderGraphTextureHandle& shadowMask);

	void DenoiseShadowMaskSSD(
		RenderGraph& renderGraph,
		RenderGraphBlackboard& blackboard,
		const SceneView& view,
		RenderGraphTextureHandle& filteredShadowMask,
		RenderGraphTextureHandle& shadowMask);

	// Experimental
	void SVGF(
		RenderGraph& renderGraph,
		RenderGraphBlackboard& blackboard,
		const SceneView& view,
		RenderGraphTextureHandle motionVectors,
		RenderGraphTextureHandle illumination,
		RenderGraphTextureHandle& svgfFilteredIllumination);

	MemoryArena* allocator;
	UIRenderer* uiRenderer;
	RenderBackend* renderBackend;
	ShaderCompiler* shaderCompiler;
	uint32 deviceMask = ~0u;

	bool renderBRDFLut = true;
	uint32 brdfLutSize = 256;
	RenderBackendTextureHandle brdfLut;

	uint32 enviromentMapSize = 1024;
	RenderBackendTextureHandle equirectangularMap;
	RenderBackendTextureHandle enviromentMap;
	RenderBackendTextureHandle iblDiffuseLut;
	RenderBackendTextureHandle iblSpecularLut;
	
	RenderGraphPersistentTexture historyBufferCache;
	RenderGraphPersistentTexture historyDepthBufferCache;
	RenderGraphPersistentTexture prevLinearDepthBufferRB;
	RenderGraphPersistentTexture prevIllumRB;
	RenderGraphPersistentTexture prevMomentsRB;
	RenderGraphPersistentTexture prevHistoryLengthRB;
	RenderGraphPersistentTexture prevGBuffer1RB;

	RenderBackendSamplerHandle samplerLinearClamp;
	RenderBackendSamplerHandle samplerLinearWarp;
	RenderBackendSamplerHandle samplerComparisonGreaterLinearClamp;

	RenderBackendBufferHandle perFrameDataBuffer;
	RenderBackendBufferHandle shadowMapParameters;

	RenderBackendShaderHandle shadowMapShader;
	RenderBackendShaderHandle screenSpaceShadowsShader;
	RenderBackendShaderHandle ssdSpatialFilteringShader;
	RenderBackendShaderHandle brdfLutShader;
	RenderBackendShaderHandle gbufferShader;
	RenderBackendShaderHandle motionVectorsShader;
	RenderBackendShaderHandle lightingShader;
	RenderBackendShaderHandle dofShader;
	RenderBackendShaderHandle tonemappingShader;
	RenderBackendShaderHandle fxaaShader;
	RenderBackendShaderHandle gtaoMainShader;
	RenderBackendShaderHandle skyBoxShader;

	RenderBackendRayTracingPipelineStateHandle rayTracingShadowsPipelineState;
	RenderBackendBufferHandle rayTracingShadowsSBT;

	RenderBackendShaderHandle svgfReprojectCS;
	RenderBackendShaderHandle svgfFilterMomentsCS;
	RenderBackendShaderHandle svgfAtrousCS;
	RenderBackendShaderHandle captureFrameCS;

	SkyAtmosphere* skyAtmosphere;
};

}