#include "PostProcessingCommon.h"
#include "HybridRenderPipeline.h"

#include <optick.h>

import HorizonEngine.Entity;

namespace HE
{

#define CREATE_SHADER(shader, filename, entry, stage) { RenderBackendShaderDesc shaderDesc = {};LoadShaderSourceFromFile(filename, source); CompileShader(shaderCompiler,source,HE_TEXT(entry),stage,ShaderRepresentation::SPIRV,includeDirs,defines,&shaderDesc.stages[(uint32)stage]); shaderDesc.entryPoints[(uint32)stage] = entry; shader = RenderBackendCreateShader(renderBackend, deviceMask, &shaderDesc, "shader");}

static const HybridRenderPipelineSettings hybridRenderPipelineSettings = {

};

HybridRenderPipeline::HybridRenderPipeline(RenderContext* context)
	: allocator(context->arena)
	, renderBackend(context->renderBackend)
	, shaderCompiler(context->shaderCompiler)
	, uiRenderer(context->uiRenderer)
{

}

void HybridRenderPipeline::Init()
{
	RenderBackendSamplerDesc samplerLinearClampDesc = RenderBackendSamplerDesc::CreateLinearClamp(0.0f, -FLOAT_MAX, FLOAT_MAX, 1);
	samplerLinearClamp = RenderBackendCreateSampler(renderBackend, deviceMask, &samplerLinearClampDesc, "SamplerLinearClamp");
	RenderBackendSamplerDesc samplerLinearWarpDesc = RenderBackendSamplerDesc::CreateLinearWarp(0.0f, -FLOAT_MAX, FLOAT_MAX, 1);
	samplerLinearWarp = RenderBackendCreateSampler(renderBackend, deviceMask, &samplerLinearWarpDesc, "SamplerLinearWarp");
	RenderBackendSamplerDesc samplerComparisonGreaterLinearClampDesc = RenderBackendSamplerDesc::CreateComparisonLinearClamp(0.0f, -FLOAT_MAX, FLOAT_MAX, 1, CompareOp::Greater);
	samplerComparisonGreaterLinearClamp = RenderBackendCreateSampler(renderBackend, deviceMask, &samplerComparisonGreaterLinearClampDesc, "SamplerComparisonGreaterLinearClamp");

	RenderBackendTextureDesc brdfLutDesc = RenderBackendTextureDesc::Create2D(brdfLutSize, brdfLutSize, PixelFormat::RG16Float, TextureCreateFlags::UnorderedAccess | TextureCreateFlags::ShaderResource);
	brdfLut = RenderBackendCreateTexture(renderBackend, deviceMask, &brdfLutDesc, nullptr, "BRDFLut");

	RenderBackendBufferDesc perFrameDataBufferDesc = RenderBackendBufferDesc::CreateByteAddress(sizeof(PerFrameData));
	perFrameDataBuffer = RenderBackendCreateBuffer(renderBackend, deviceMask, &perFrameDataBufferDesc, "PerFrameDataBuffer");

	RenderBackendBufferDesc shadowMapParametersDesc = RenderBackendBufferDesc::CreateByteAddress(sizeof(ShadowMapShaderParameters));
	shadowMapParameters = RenderBackendCreateBuffer(renderBackend, deviceMask, &shadowMapParametersDesc, "shadowMapParamaters");

	RenderGraphTextureDesc historyBufferDesc = RenderGraphTextureDesc::Create2D(
		1920,
		1080,
		PixelFormat::BGRA8Unorm,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess);
	historyBufferCache.texture = RenderBackendCreateTexture(renderBackend, deviceMask, &historyBufferDesc, nullptr, "HistoryBuffer");
	historyBufferCache.desc = historyBufferDesc;
	historyBufferCache.initialState = RenderBackendResourceState::ShaderResource;

	RenderGraphTextureDesc prevLinearDepthBufferRBDesc = RenderGraphTextureDesc::Create2D(
		1920,
		1080,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::RenderTarget);
	prevLinearDepthBufferRB.texture = RenderBackendCreateTexture(renderBackend, deviceMask, &prevLinearDepthBufferRBDesc, nullptr, "prevLinearDepthBufferRB");
	prevLinearDepthBufferRB.desc = prevLinearDepthBufferRBDesc;
	prevLinearDepthBufferRB.initialState = RenderBackendResourceState::ShaderResource;

	RenderGraphTextureDesc prevIllumRBDesc = RenderGraphTextureDesc::Create2D(
		1920,
		1080,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess);
	prevIllumRB.texture = RenderBackendCreateTexture(renderBackend, deviceMask, &prevIllumRBDesc, nullptr, "prevIllumRB");
	prevIllumRB.desc = prevIllumRBDesc;
	prevIllumRB.initialState = RenderBackendResourceState::UnorderedAccess;

	RenderGraphTextureDesc prevMomentsRBDesc = RenderGraphTextureDesc::Create2D(
		1920,
		1080,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess);
	prevMomentsRB.texture = RenderBackendCreateTexture(renderBackend, deviceMask, &prevMomentsRBDesc, nullptr, "prevMomentsRB");
	prevMomentsRB.desc = prevMomentsRBDesc;
	prevMomentsRB.initialState = RenderBackendResourceState::UnorderedAccess;

	RenderGraphTextureDesc prevHistoryLengthRBDesc = RenderGraphTextureDesc::Create2D(
		1920,
		1080,
		PixelFormat::R16Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess);
	prevHistoryLengthRB.texture = RenderBackendCreateTexture(renderBackend, deviceMask, &prevHistoryLengthRBDesc, nullptr, "prevHistoryLengthRB");
	prevHistoryLengthRB.desc = prevHistoryLengthRBDesc;
	prevHistoryLengthRB.initialState = RenderBackendResourceState::UnorderedAccess;
	
	RenderGraphTextureDesc prevGBuffer1RBDesc = RenderGraphTextureDesc::Create2D(
		1920,
		1080,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::RenderTarget);
	prevGBuffer1RB.texture = RenderBackendCreateTexture(renderBackend, deviceMask, &prevGBuffer1RBDesc, nullptr, "prevGBuffer1RB");
	prevGBuffer1RB.desc = prevGBuffer1RBDesc;
	prevGBuffer1RB.initialState = RenderBackendResourceState::ShaderResource;

	std::vector<uint8> source;
	std::vector<const wchar*> includeDirs;
	std::vector<const wchar*> defines;
	includeDirs.push_back(HE_TEXT("../../../Shaders"));
	includeDirs.push_back(HE_TEXT("../../../Shaders/HybridRenderPipeline"));
	
	RenderBackendShaderDesc skyBoxShaderDesc;
	skyBoxShaderDesc.rasterizationState.cullMode = RasterizationCullMode::None;
	skyBoxShaderDesc.rasterizationState.frontFaceCounterClockwise = true;
	skyBoxShaderDesc.colorBlendState.numColorAttachments = 1;
	skyBoxShaderDesc.depthStencilState = {
		.depthTestEnable = true,
		.depthWriteEnable = false,
		.depthCompareOp = CompareOp::Equal,
		.stencilTestEnable = false,
	};
	LoadShaderSourceFromFile("../../../Shaders/HybridRenderPipeline/SkyBox.hsf", source);
	CompileShader(
		shaderCompiler,
		source,
		HE_TEXT("SkyBoxVS"),
		RenderBackendShaderStage::Vertex,
		ShaderRepresentation::SPIRV,
		includeDirs,
		defines,
		&skyBoxShaderDesc.stages[(uint32)RenderBackendShaderStage::Vertex]);
	skyBoxShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Vertex] = "SkyBoxVS";
	CompileShader(
		shaderCompiler,
		source,
		HE_TEXT("SkyBoxPS"),
		RenderBackendShaderStage::Pixel,
		ShaderRepresentation::SPIRV,
		includeDirs,
		defines,
		&skyBoxShaderDesc.stages[(uint32)RenderBackendShaderStage::Pixel]);
	skyBoxShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Pixel] = "SkyBoxPS";
	skyBoxShader = RenderBackendCreateShader(renderBackend, deviceMask, &skyBoxShaderDesc, "SkyBoxShader");

	
	RenderBackendShaderDesc gbufferShaderDesc;
	gbufferShaderDesc.rasterizationState.cullMode = RasterizationCullMode::None;
	gbufferShaderDesc.rasterizationState.frontFaceCounterClockwise = true;
	gbufferShaderDesc.colorBlendState.numColorAttachments = 6;
	gbufferShaderDesc.depthStencilState = {
		.depthTestEnable = true,
		.depthWriteEnable = true,
		.depthCompareOp = CompareOp::GreaterOrEqual,
		.stencilTestEnable = false,
	};
	LoadShaderSourceFromFile("../../../Shaders/HybridRenderPipeline/GBuffer.hsf", source);
	CompileShader(
		shaderCompiler,
		source,
		HE_TEXT("GBufferVS"),
		RenderBackendShaderStage::Vertex,
		ShaderRepresentation::SPIRV,
		includeDirs,
		defines,
		&gbufferShaderDesc.stages[(uint32)RenderBackendShaderStage::Vertex]);
	gbufferShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Vertex] = "GBufferVS";
	CompileShader(
		shaderCompiler, 
		source,
		HE_TEXT("GBufferPS"),
		RenderBackendShaderStage::Pixel,
		ShaderRepresentation::SPIRV,
		includeDirs,
		defines,
		&gbufferShaderDesc.stages[(uint32)RenderBackendShaderStage::Pixel]);
	gbufferShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Pixel] = "GBufferPS";
	gbufferShader = RenderBackendCreateShader(renderBackend, deviceMask, &gbufferShaderDesc, "GBufferShader");

	RenderBackendShaderDesc shadowMapShaderDesc;
	shadowMapShaderDesc.rasterizationState.cullMode = RasterizationCullMode::None;
	shadowMapShaderDesc.rasterizationState.frontFaceCounterClockwise = true;
	shadowMapShaderDesc.colorBlendState.numColorAttachments = 0;
	shadowMapShaderDesc.depthStencilState = {
		.depthTestEnable = true,
		.depthWriteEnable = true,
		.depthCompareOp = CompareOp::GreaterOrEqual,
		.stencilTestEnable = false,
	};
	LoadShaderSourceFromFile("../../../Shaders/HybridRenderPipeline/ShadowMap.hsf", source);
	CompileShader(
		shaderCompiler,
		source,
		HE_TEXT("ShadowMapVS"),
		RenderBackendShaderStage::Vertex,
		ShaderRepresentation::SPIRV,
		includeDirs,
		defines,
		&shadowMapShaderDesc.stages[(uint32)RenderBackendShaderStage::Vertex]);
	shadowMapShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Vertex] = "ShadowMapVS";
	CompileShader(
		shaderCompiler,
		source,
		HE_TEXT("ShadowMapPS"),
		RenderBackendShaderStage::Pixel,
		ShaderRepresentation::SPIRV,
		includeDirs,
		defines,
		&shadowMapShaderDesc.stages[(uint32)RenderBackendShaderStage::Pixel]);
	shadowMapShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Pixel] = "ShadowMapPS";
	shadowMapShader = RenderBackendCreateShader(renderBackend, deviceMask, &shadowMapShaderDesc, "ShadowMapShader");

	CREATE_SHADER(brdfLutShader, "../../../Shaders/BRDFLut.hsf", "BRDFLutCS", RenderBackendShaderStage::Compute);
	CREATE_SHADER(motionVectorsShader, "../../../Shaders/HybridRenderPipeline/MotionVectors.hsf", "MotionVectorsCS", RenderBackendShaderStage::Compute);
	CREATE_SHADER(screenSpaceShadowsShader, "../../../Shaders/HybridRenderPipeline/ScreenSpaceShadows.hsf", "ScreenSpaceShadowsCS", RenderBackendShaderStage::Compute);
	CREATE_SHADER(gtaoMainShader, "../../../Shaders/HybridRenderPipeline/AmbientOcclusion.hsf", "GTAOHorizonSearchIntegralCS", RenderBackendShaderStage::Compute);;
	CREATE_SHADER(lightingShader, "../../../Shaders/HybridRenderPipeline/Lighting.hsf", "LightingCS", RenderBackendShaderStage::Compute);
	CREATE_SHADER(tonemappingShader, "../../../Shaders/HybridRenderPipeline/Tonemapping.hsf", "TonemappingCS", RenderBackendShaderStage::Compute);
	CREATE_SHADER(fxaaShader, "../../../Shaders/HybridRenderPipeline/FXAA.hsf", "FxaaCS", RenderBackendShaderStage::Compute);
	CREATE_SHADER(ssdSpatialFilteringShader, "../../../Shaders/ScreenSpaceDenoising/SSDSpatialFiltering.hsf", "SDDSpatialFilteringCS", RenderBackendShaderStage::Compute);

	CREATE_SHADER(svgfReprojectCS, "../../../Shaders/HybridRenderPipeline/SVGF.hsf", "SVGFReprojectCS", RenderBackendShaderStage::Compute);
	CREATE_SHADER(svgfFilterMomentsCS, "../../../Shaders/HybridRenderPipeline/SVGF.hsf", "SVGFFilterMomentsCS", RenderBackendShaderStage::Compute);
	CREATE_SHADER(svgfAtrousCS, "../../../Shaders/HybridRenderPipeline/SVGF.hsf", "SVGFAtrousCS", RenderBackendShaderStage::Compute);

#if DEBUG_ONLY_RAY_TRACING_ENBALE
	RenderBackendRayTracingPipelineStateDesc rayTracingShadowsPipelineStateDesc = {
		.maxRayRecursionDepth = 1,
	};

	LoadShaderSourceFromFile("../../../Shaders/HybridRenderPipeline/RayTracingShadows.hsf", source);

	rayTracingShadowsPipelineStateDesc.shaders.push_back(RenderBackendRayTracingShaderDesc{
		.stage = RenderBackendShaderStage::RayGen,
		.entry = "RayTracingShadowsRayGen",
	});
	CompileShader(
		shaderCompiler,
		source,
		HE_TEXT("RayTracingShadowsRayGen"),
		RenderBackendShaderStage::RayGen,
		ShaderRepresentation::SPIRV,
		includeDirs,
		defines,
		&rayTracingShadowsPipelineStateDesc.shaders[0].code);

	rayTracingShadowsPipelineStateDesc.shaders.push_back(RenderBackendRayTracingShaderDesc{
		.stage = RenderBackendShaderStage::Miss,
		.entry = "RayTracingShadowsMiss",
	});
	CompileShader(
		shaderCompiler,
		source,
		HE_TEXT("RayTracingShadowsMiss"),
		RenderBackendShaderStage::Miss,
		ShaderRepresentation::SPIRV,
		includeDirs,
		defines,
		&rayTracingShadowsPipelineStateDesc.shaders[1].code);

	rayTracingShadowsPipelineStateDesc.shaderGroupDescs.resize(2);
	rayTracingShadowsPipelineStateDesc.shaderGroupDescs[0] = RenderBackendRayTracingShaderGroupDesc::CreateRayGen(0);
	rayTracingShadowsPipelineStateDesc.shaderGroupDescs[1] = RenderBackendRayTracingShaderGroupDesc::CreateMiss(1);

	rayTracingShadowsPipelineState = RenderBackendCreateRayTracingPipelineState(renderBackend, deviceMask, &rayTracingShadowsPipelineStateDesc, "rayTracingShadowsPipelineState");
	
	RenderBackendRayTracingShaderBindingTableDesc rayTracingShadowsSBTDesc = {
		.rayTracingPipelineState = rayTracingShadowsPipelineState,
		.numShaderRecords = 0,
	};
	rayTracingShadowsSBT = RenderBackendCreateRayTracingShaderBindingTable(renderBackend, deviceMask, &rayTracingShadowsSBTDesc, "rayTracingShadowsSBT");
#endif


#if 0
	SkyAtmosphereConfig config;
	skyAtmosphere = CreateSkyAtmosphere(renderBackend, shaderCompiler, &config);
#endif
}

void CooyRenderGraphFinalTextureToCameraTarget(RenderGraph* renderGraph)
{
	RenderGraphBlackboard& blackboard = renderGraph->blackboard;

	renderGraph->AddPass("PresentPass", RenderGraphPassFlags::NeverGetCulled,
	[&](RenderGraphBuilder& builder)
	{
		const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
		const auto& finalTextureData = blackboard.Get<RenderGraphFinalTexture>();
		auto& outputTextureData = blackboard.Get<RenderGraphOutputTexture>();

		auto finalTexture = builder.ReadTexture(finalTextureData.finalTexture, RenderBackendResourceState::CopySrc);
		auto outputTexture = outputTextureData.outputTexture = builder.WriteTexture(outputTextureData.outputTexture, RenderBackendResourceState::CopyDst);

		return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
		{
			Offset2D offset = {};
			Extent2D extent = { perFrameData.data.targetResolutionWidth, perFrameData.data.targetResolutionHeight };
			commandList.CopyTexture2D(
				registry.GetRenderBackendTexture(finalTexture),
				offset,
				0,
				registry.GetRenderBackendTexture(outputTexture),
				offset,
				0,
				extent);
			RenderBackendBarrier transition = RenderBackendBarrier(registry.GetRenderBackendTexture(outputTexture), RenderBackendTextureSubresourceRange(0, 1, 0, 1), RenderBackendResourceState::CopyDst, RenderBackendResourceState::Present);
			commandList.Transitions(&transition, 1);
		};
	});
}

void HybridRenderPipeline::RenderRayTracingShadows(
	RenderGraph& renderGraph,
	RenderGraphBlackboard& blackboard,
	const SceneView& view,
	const LightRenderProxy& lightProxy,
	RenderBackendBufferHandle parametersBuffer,
	RenderGraphTextureHandle& shadowMask)
{
	renderGraph.AddPass("RayTracingShadowsPass", RenderGraphPassFlags::RayTrace,
		[&](RenderGraphBuilder& builder)
		{
			const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
			const auto& depthBufferData = blackboard.Get<RenderGraphDepthBuffer>();

			auto depthBuffer = builder.ReadTexture(depthBufferData.depthBuffer, RenderBackendResourceState::ShaderResource);

			shadowMask = builder.WriteTexture(shadowMask, RenderBackendResourceState::UnorderedAccess);

			return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
			{
				uint32 width = perFrameData.data.renderResolutionWidth;
				uint32 height = perFrameData.data.renderResolutionHeight;

				ShaderArguments shaderArguments = {};
				shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
				shaderArguments.BindAS(1, view.scene->topLevelAS);
				shaderArguments.BindTextureSRV(2, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(depthBuffer)));
				shaderArguments.BindTextureUAV(3, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(shadowMask), 0));

				commandList.TraceRays(
					rayTracingShadowsPipelineState,
					rayTracingShadowsSBT,
					shaderArguments,
					width,
					height,
					1);
			};
		});
}

void HybridRenderPipeline::RenderScreenSpaceShadows(
	RenderGraph& renderGraph,
	RenderGraphBlackboard& blackboard,
	const SceneView& view,
	const LightRenderProxy& lightProxy,
	RenderBackendBufferHandle parametersBuffer,
	RenderGraphTextureHandle& shadowMask)
{
	uint32 numDynamicShadowCascades = lightProxy.GetNumDynamicShadowCascades();
	uint32 shadowMapSize = lightProxy.GetShadowMapSize();

	RenderGraphTextureDesc shadowMapDesc = RenderGraphTextureDesc::Create2DArray(
		shadowMapSize,
		shadowMapSize,
		PixelFormat::D32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::DepthStencil,
		numDynamicShadowCascades,
		RenderTargetClearValue::CreateDepthValue(0.0f));

	auto shadowMap = renderGraph.CreateTexture(shadowMapDesc, "ShadowMap");

	ShadowCascades cascades;
	LightRenderProxy::CalculateShadowCascades(lightProxy, view.camera, cascades);
	ShadowMapShaderParameters parameters = {};
	for (uint32 cascadeIndex = 0; cascadeIndex < numDynamicShadowCascades; cascadeIndex++)
	{
		parameters.viewProjectionMatrix[cascadeIndex] = cascades.viewProjectionMatrix[cascadeIndex];
		parameters.splitDepth[cascadeIndex] = cascades.splitDepth[cascadeIndex];
		parameters.shadowMapInvSize = 1.0f / shadowMapSize;
	}

	void* data = nullptr;
	RenderBackendMapBuffer(renderBackend, parametersBuffer, &data);
	memcpy((uint8*)data, &parameters, sizeof(ShadowMapShaderParameters));
	RenderBackendUnmapBuffer(renderBackend, parametersBuffer);

	for (uint32 cascadeIndex = 0; cascadeIndex < numDynamicShadowCascades; cascadeIndex++)
	{
		renderGraph.AddPass("ShadowMapPass", RenderGraphPassFlags::Raster,
			[&](RenderGraphBuilder& builder)
			{
				const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();

				shadowMap = builder.WriteTexture(shadowMap, RenderBackendResourceState::DepthStencil);

				builder.BindDepthTarget(shadowMap, RenderTargetLoadOp::Clear, RenderTargetStoreOp::Store, 0, cascadeIndex);

				return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
				{
					//commandList.UpdateBuffer(parametersBuffer, 0, &parameters, sizeof(ShadowMapShaderParameters));

					RenderBackendViewport viewport(0.0f, 0.0f, (float)shadowMapSize, (float)shadowMapSize);
					RenderBackendScissor scissor(0, 0, shadowMapSize, shadowMapSize);

					commandList.SetViewports(&viewport, 1);
					commandList.SetScissors(&scissor, 1);

					for (const auto& renderable : view.scene->renderables)
					{
						RenderBackendBufferHandle vertexPosition = view.scene->vertexBuffers[0][renderable.vertexBufferIndex];
						RenderBackendBufferHandle indexBuffer = view.scene->indexBuffers[renderable.indexBufferIndex];
						RenderBackendBufferHandle worldMatrixBuffer = view.scene->worldMatrixBuffer;

						ShaderArguments shaderArguments = {};
						shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
						shaderArguments.BindBuffer(1, vertexPosition, 0);
						shaderArguments.BindBuffer(2, worldMatrixBuffer, renderable.transformIndex);
						shaderArguments.BindBuffer(3, shadowMapParameters, 0);
						shaderArguments.PushConstants(0, 1.0f * cascadeIndex);

						commandList.DrawIndexed(
							shadowMapShader,
							shaderArguments,
							indexBuffer,
							renderable.numIndices,
							1,
							renderable.firstIndex,
							0,
							0,
							PrimitiveTopology::TriangleList);
					}

					viewport = RenderBackendViewport(0.0f, 0.0f, (float)perFrameData.data.renderResolutionWidth, (float)perFrameData.data.renderResolutionHeight);
					scissor = RenderBackendScissor(0, 0, perFrameData.data.renderResolutionWidth, perFrameData.data.renderResolutionHeight);

					commandList.SetScissors(&scissor, 1);
					commandList.SetViewports(&viewport, 1);
				};
			});
	}

	renderGraph.AddPass("ScreenSpaceShadowsPass", RenderGraphPassFlags::Compute,
		[&](RenderGraphBuilder& builder)
		{
			const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
			const auto& gbufferData = blackboard.Get<RenderGraphGBuffer>();
			const auto& depthBufferData = blackboard.Get<RenderGraphDepthBuffer>();
			auto gbuffer0 = builder.ReadTexture(gbufferData.gbuffer0, RenderBackendResourceState::ShaderResource);
			auto depthBuffer = builder.ReadTexture(depthBufferData.depthBuffer, RenderBackendResourceState::ShaderResource);
			shadowMap = builder.ReadTexture(shadowMap, RenderBackendResourceState::ShaderResource);
			shadowMask = builder.WriteTexture(shadowMask, RenderBackendResourceState::UnorderedAccess);

			return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
			{
				uint32 dispatchWidth = CEIL_DIV(perFrameData.data.renderResolutionWidth, 8);
				uint32 dispatchHeight = CEIL_DIV(perFrameData.data.renderResolutionHeight, 8);

				ShaderArguments shaderArguments = {};
				shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
				shaderArguments.BindBuffer(4, shadowMapParameters, 0);
				shaderArguments.BindTextureSRV(1, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(gbuffer0)));
				shaderArguments.BindTextureSRV(2, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(depthBuffer)));
				shaderArguments.BindTextureSRV(3, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(shadowMap)));
				shaderArguments.BindTextureUAV(5, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(shadowMask), 0));

				commandList.Dispatch2D(
					screenSpaceShadowsShader,
					shaderArguments,
					dispatchWidth,
					dispatchHeight);
			};
		});
}

void HybridRenderPipeline::DenoiseShadowMaskSSD(
	RenderGraph& renderGraph,
	RenderGraphBlackboard& blackboard,
	const SceneView& view,
	RenderGraphTextureHandle& filteredShadowMask,
	RenderGraphTextureHandle& shadowMask)
{
	renderGraph.AddPass("SSDSpatialFilteringPass1", RenderGraphPassFlags::Compute,
		[&](RenderGraphBuilder& builder)
		{
			const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
			const auto& gbufferData = blackboard.Get<RenderGraphGBuffer>();
			const auto& depthBufferData = blackboard.Get<RenderGraphDepthBuffer>();
			auto gbuffer0 = builder.ReadTexture(gbufferData.gbuffer0, RenderBackendResourceState::ShaderResource);
			shadowMask = builder.ReadTexture(shadowMask, RenderBackendResourceState::ShaderResource);
			auto depthBuffer = builder.ReadTexture(depthBufferData.depthBuffer, RenderBackendResourceState::ShaderResource);

			filteredShadowMask = builder.WriteTexture(filteredShadowMask, RenderBackendResourceState::UnorderedAccess);

			return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
			{
				uint32 dispatchWidth = CEIL_DIV(perFrameData.data.renderResolutionWidth, 8);
				uint32 dispatchHeight = CEIL_DIV(perFrameData.data.renderResolutionHeight, 8);

				ShaderArguments shaderArguments = {};
				shaderArguments.BindTextureSRV(0, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(shadowMask)));
				shaderArguments.BindTextureSRV(1, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(depthBuffer)));
				shaderArguments.BindTextureUAV(2, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(filteredShadowMask), 0));
				shaderArguments.PushConstants(0, 1.0);

				commandList.Dispatch2D(
					ssdSpatialFilteringShader,
					shaderArguments,
					dispatchWidth,
					dispatchHeight);
			};
		});

	renderGraph.AddPass("SSDSpatialFilteringPass2", RenderGraphPassFlags::Compute,
		[&](RenderGraphBuilder& builder)
		{
			const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
			const auto& gbufferData = blackboard.Get<RenderGraphGBuffer>();
			const auto& depthBufferData = blackboard.Get<RenderGraphDepthBuffer>();
			auto gbuffer0 = builder.ReadTexture(gbufferData.gbuffer0, RenderBackendResourceState::ShaderResource);
			filteredShadowMask = builder.ReadTexture(filteredShadowMask, RenderBackendResourceState::ShaderResource);
			auto depthBuffer = builder.ReadTexture(depthBufferData.depthBuffer, RenderBackendResourceState::ShaderResource);

			shadowMask = builder.WriteTexture(shadowMask, RenderBackendResourceState::UnorderedAccess);

			return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
			{
				uint32 dispatchWidth = CEIL_DIV(perFrameData.data.renderResolutionWidth, 8);
				uint32 dispatchHeight = CEIL_DIV(perFrameData.data.renderResolutionHeight, 8);

				ShaderArguments shaderArguments = {};
				shaderArguments.BindTextureSRV(0, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(filteredShadowMask)));
				shaderArguments.BindTextureSRV(1, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(depthBuffer)));
				shaderArguments.BindTextureUAV(2, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(shadowMask), 0));
				shaderArguments.PushConstants(0, -1.0);

				commandList.Dispatch2D(
					ssdSpatialFilteringShader,
					shaderArguments,
					dispatchWidth,
					dispatchHeight);
			};
		});
}

void HybridRenderPipeline::SVGF(
	RenderGraph& renderGraph, 
	RenderGraphBlackboard& blackboard, 
	const SceneView& view, 
	RenderGraphTextureHandle motionVectors, 
	RenderGraphTextureHandle illumination,
	RenderGraphTextureHandle& svgfFilteredIllumination)
{
	/*
	auto svgfFilteredIllumination = renderGraph->CreateTexture(RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess),
		"SVGFFilteredIllumination");
	*/

	auto& perFrameData = blackboard.CreateSingleton<RenderGraphPerFrameData>();
	auto& gbufferData = blackboard.CreateSingleton<RenderGraphGBuffer>();

	float filterIterations = 4;
	float feedbackTap = 1;
	float phiColor = 10.0;
	float phiNormal = 128.0;
	float alpha = 0.05;
	float momentsAlpha = 0.2;

	auto svgfIllumination = renderGraph.CreateTexture(RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess),
		"SVGFIllumination");

	auto svgfMoments = renderGraph.CreateTexture(RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess),
		"SVGFMoments");

	auto svgfHistoryLength = renderGraph.CreateTexture(RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::R16Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess),
		"SVGFHistoryLength");

	RenderGraphTextureHandle prevGBuffer1 = renderGraph.ImportExternalTexture(prevGBuffer1RB.texture, prevGBuffer1RB.desc, prevGBuffer1RB.initialState, "prevGBuffer1RB");
	RenderGraphTextureHandle prevLinearDepthBuffer = renderGraph.ImportExternalTexture(prevLinearDepthBufferRB.texture, prevLinearDepthBufferRB.desc, prevLinearDepthBufferRB.initialState, "prevLinearDepthBufferRB");
	RenderGraphTextureHandle prevIllum = renderGraph.ImportExternalTexture(prevIllumRB.texture, prevIllumRB.desc, prevIllumRB.initialState, "prevIllumRB");
	RenderGraphTextureHandle prevMoments = renderGraph.ImportExternalTexture(prevMomentsRB.texture, prevMomentsRB.desc, prevMomentsRB.initialState, "prevMomentsRB");
	RenderGraphTextureHandle prevHistoryLength = renderGraph.ImportExternalTexture(prevHistoryLengthRB.texture, prevHistoryLengthRB.desc, prevHistoryLengthRB.initialState, "prevHistoryLengthRB");

	renderGraph.AddPass("SVGFReprojectPass", RenderGraphPassFlags::Compute,
		[&](RenderGraphBuilder& builder)
		{
			const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
			const auto& gbufferData = blackboard.Get<RenderGraphGBuffer>();
			const auto& depthBufferData = blackboard.Get<RenderGraphDepthBuffer>();

			illumination = builder.ReadTexture(illumination, RenderBackendResourceState::ShaderResource);
			auto gbuffer1 = builder.ReadTexture(gbufferData.gbuffer1, RenderBackendResourceState::ShaderResource);
			auto linearDepthBuffer = builder.ReadTexture(gbufferData.gbuffer4, RenderBackendResourceState::ShaderResource);
			motionVectors = builder.ReadTexture(motionVectors, RenderBackendResourceState::ShaderResource);

			prevGBuffer1 = builder.ReadTexture(prevGBuffer1, RenderBackendResourceState::ShaderResource);
			prevLinearDepthBuffer = builder.ReadTexture(prevLinearDepthBuffer, RenderBackendResourceState::ShaderResource);
			prevIllum = builder.ReadTexture(prevIllum, RenderBackendResourceState::UnorderedAccess);
			prevMoments = builder.ReadTexture(prevMoments, RenderBackendResourceState::UnorderedAccess);
			prevHistoryLength = builder.ReadTexture(prevHistoryLength, RenderBackendResourceState::UnorderedAccess);

			svgfIllumination = builder.WriteTexture(svgfIllumination, RenderBackendResourceState::UnorderedAccess);
			svgfMoments = builder.WriteTexture(svgfMoments, RenderBackendResourceState::UnorderedAccess);
			svgfHistoryLength = builder.WriteTexture(svgfHistoryLength, RenderBackendResourceState::UnorderedAccess);

			return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
			{
				uint32 dispatchWidth = CEIL_DIV(perFrameData.data.renderResolutionWidth, 8);
				uint32 dispatchHeight = CEIL_DIV(perFrameData.data.renderResolutionHeight, 8);

				ShaderArguments shaderArguments = {};
				shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
				shaderArguments.BindTextureSRV(1, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(illumination)));
				shaderArguments.BindTextureSRV(2, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(gbuffer1)));
				shaderArguments.BindTextureSRV(3, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(linearDepthBuffer)));
				shaderArguments.BindTextureSRV(4, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(motionVectors)));
				shaderArguments.BindTextureUAV(5, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(svgfIllumination), 0));
				shaderArguments.BindTextureUAV(6, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(svgfMoments), 0));
				shaderArguments.BindTextureUAV(7, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(svgfHistoryLength), 0));
				shaderArguments.BindTextureSRV(8, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(prevGBuffer1)));
				shaderArguments.BindTextureSRV(9, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(prevLinearDepthBuffer)));
				shaderArguments.BindTextureUAV(10, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(prevIllum), 0));
				shaderArguments.BindTextureUAV(11, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(prevMoments), 0));
				shaderArguments.BindTextureUAV(12, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(prevHistoryLength), 0));
				shaderArguments.PushConstants(0, phiColor);
				shaderArguments.PushConstants(1, phiNormal);
				shaderArguments.PushConstants(2, alpha);
				shaderArguments.PushConstants(3, momentsAlpha);

				commandList.Dispatch2D(
					svgfReprojectCS,
					shaderArguments,
					dispatchWidth,
					dispatchHeight);
			};
		});

	renderGraph.AddPass("SVFGFilterMomentsPass", RenderGraphPassFlags::Compute,
		[&](RenderGraphBuilder& builder)
		{
			const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
			const auto& gbufferData = blackboard.Get<RenderGraphGBuffer>();
			const auto& depthBufferData = blackboard.Get<RenderGraphDepthBuffer>();

			auto gbuffer1 = builder.ReadTexture(gbufferData.gbuffer1, RenderBackendResourceState::ShaderResource);
			auto linearDepthBuffer = builder.ReadTexture(gbufferData.gbuffer4, RenderBackendResourceState::ShaderResource);
			svgfIllumination = builder.ReadTexture(svgfIllumination, RenderBackendResourceState::UnorderedAccess);
			svgfMoments = builder.ReadTexture(svgfMoments, RenderBackendResourceState::UnorderedAccess);
			svgfHistoryLength = builder.ReadTexture(svgfHistoryLength, RenderBackendResourceState::UnorderedAccess);

			svgfFilteredIllumination = builder.WriteTexture(svgfFilteredIllumination, RenderBackendResourceState::UnorderedAccess);

			return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
			{
				uint32 dispatchWidth = CEIL_DIV(perFrameData.data.renderResolutionWidth, 8);
				uint32 dispatchHeight = CEIL_DIV(perFrameData.data.renderResolutionHeight, 8);

				ShaderArguments shaderArguments = {};
				shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
				shaderArguments.BindTextureSRV(2, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(gbuffer1)));
				shaderArguments.BindTextureSRV(3, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(linearDepthBuffer)));
				shaderArguments.BindTextureUAV(5, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(svgfIllumination), 0));
				shaderArguments.BindTextureUAV(6, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(svgfMoments), 0));
				shaderArguments.BindTextureUAV(7, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(svgfHistoryLength), 0));
				shaderArguments.BindTextureUAV(13, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(svgfFilteredIllumination), 0));
				shaderArguments.PushConstants(0, phiColor);
				shaderArguments.PushConstants(1, phiNormal);
				shaderArguments.PushConstants(2, alpha);
				shaderArguments.PushConstants(3, momentsAlpha);

				commandList.Dispatch2D(
					svgfFilterMomentsCS,
					shaderArguments,
					dispatchWidth,
					dispatchHeight);
			};
		});

	renderGraph.AddPass("SVFGAtrousPass", RenderGraphPassFlags::RayTrace,
		[&](RenderGraphBuilder& builder)
		{
			const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
			const auto& gbufferData = blackboard.Get<RenderGraphGBuffer>();
			const auto& depthBufferData = blackboard.Get<RenderGraphDepthBuffer>();

			auto gbuffer1 = builder.ReadTexture(gbufferData.gbuffer1, RenderBackendResourceState::ShaderResource);
			auto linearDepthBuffer = builder.ReadTexture(gbufferData.gbuffer4, RenderBackendResourceState::ShaderResource);
			svgfIllumination = builder.ReadTexture(svgfIllumination, RenderBackendResourceState::UnorderedAccess);
			svgfHistoryLength = builder.ReadTexture(svgfHistoryLength, RenderBackendResourceState::UnorderedAccess);

			svgfFilteredIllumination = builder.WriteTexture(svgfFilteredIllumination, RenderBackendResourceState::UnorderedAccess);

			return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
			{
				uint32 dispatchWidth = CEIL_DIV(perFrameData.data.renderResolutionWidth, 8);
				uint32 dispatchHeight = CEIL_DIV(perFrameData.data.renderResolutionHeight, 8);

				for (uint32 iterationIndex = 0; iterationIndex < filterIterations; iterationIndex++)
				{
					float stepSize = (float)(1 << iterationIndex);

					uint32 width = perFrameData.data.renderResolutionWidth;
					uint32 height = perFrameData.data.renderResolutionHeight;

					ShaderArguments shaderArguments = {};
					shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
					shaderArguments.BindTextureSRV(2, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(gbuffer1)));
					shaderArguments.BindTextureSRV(3, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(linearDepthBuffer)));
					shaderArguments.BindTextureUAV(5, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(svgfIllumination), 0));
					shaderArguments.BindTextureUAV(7, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(svgfHistoryLength), 0));
					shaderArguments.BindTextureUAV(13, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(svgfFilteredIllumination), 0));
					shaderArguments.PushConstants(0, phiColor);
					shaderArguments.PushConstants(1, phiNormal);
					shaderArguments.PushConstants(2, alpha);
					shaderArguments.PushConstants(3, momentsAlpha);
					shaderArguments.PushConstants(4, stepSize);

					commandList.Dispatch2D(
						svgfAtrousCS,
						shaderArguments,
						dispatchWidth,
						dispatchHeight);
				}
			};
		});

	renderGraph.ExportTextureDeferred(gbufferData.gbuffer1, &prevGBuffer1RB);
	renderGraph.ExportTextureDeferred(gbufferData.gbuffer4, &prevLinearDepthBufferRB);
	renderGraph.ExportTextureDeferred(svgfIllumination, &prevIllumRB);
	renderGraph.ExportTextureDeferred(svgfMoments, &prevMomentsRB);
	renderGraph.ExportTextureDeferred(svgfHistoryLength, &prevHistoryLengthRB);
}

void HybridRenderPipeline::SetupRenderGraph(SceneView* view, RenderGraph* renderGraph)
{
	OPTICK_EVENT();

	RenderGraphBlackboard& blackboard = renderGraph->blackboard;
	auto& perFrameData = blackboard.CreateSingleton<RenderGraphPerFrameData>();
	auto& gbufferData = blackboard.CreateSingleton<RenderGraphGBuffer>();
	auto& depthBufferData = blackboard.CreateSingleton<RenderGraphDepthBuffer>();
	auto& sceneColorData = blackboard.CreateSingleton<RenderGraphSceneColor>();
	auto& finalTextureData = blackboard.CreateSingleton<RenderGraphFinalTexture>();
	auto& ouptutTextureData = blackboard.CreateSingleton<RenderGraphOutputTexture>();

	auto prevViewProjectionMatrix = perFrameData.data.viewProjectionMatrix;
	perFrameData.data = {
		.frameIndex = view->frameIndex,
		.gamma = 2.2,
		.exposure = 1.4,
		.sunDirection = { 0.00, 0.90045, 0.43497 },
		.solarIrradiance = { 10.0f, 10.0f, 10.0f },
		.solarAngularRadius = 0.004675f,
		.sunIlluminanceScale = { 10.0, 10.0, 10.0},
		.cameraPosition = view->camera.position,
		.nearPlane = view->camera.zNear,
		.farPlane = view->camera.zFar,
		.viewMatrix = view->camera.viewMatrix,
		.invViewMatrix = view->camera.invViewMatrix,
		.projectionMatrix = view->camera.projectionMatrix,
		.invProjectionMatrix = view->camera.invProjectionMatrix,
		.viewProjectionMatrix = view->camera.projectionMatrix * view->camera.viewMatrix,
		.invViewProjectionMatrix = view->camera.invViewMatrix * view->camera.invProjectionMatrix,
		.prevViewProjectionMatrix = prevViewProjectionMatrix,
		.renderResolutionWidth = view->targetWidth,
		.renderResolutionHeight = view->targetHeight,
		.targetResolutionWidth = view->targetWidth,
		.targetResolutionHeight = view->targetHeight,
	};

	const LightRenderProxy& mainLight = view->scene->GetMainLight();
	perFrameData.data.sunDirection = mainLight.GetDirection();

	perFrameData.buffer = perFrameDataBuffer;

	void* perFrameDataBufferData = nullptr;
	RenderBackendMapBuffer(renderBackend, perFrameDataBuffer, &perFrameDataBufferData);
	memcpy((uint8*)perFrameDataBufferData, &perFrameData, sizeof(PerFrameData));
	RenderBackendUnmapBuffer(renderBackend, perFrameDataBuffer);

	ouptutTextureData.outputTexture = renderGraph->ImportExternalTexture(view->target, view->targetDesc, RenderBackendResourceState::Undefined, "CameraTarget");
	ouptutTextureData.outputTextureDesc = view->targetDesc;

	RenderTargetClearValue clearColor = RenderTargetClearValue::CreateColorValueFloat4(0.0f, 0.0f, 0.0f, 0.0f);
	RenderTargetClearValue clearDepth = RenderTargetClearValue::CreateDepthValue(0.0f); 

	RenderGraphTextureDesc vbuffer0Desc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::R32Uint,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::RenderTarget,
		clearColor);

	RenderGraphTextureDesc gbuffer0Desc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::RenderTarget,
		clearColor);
	gbufferData.gbuffer0 = renderGraph->CreateTexture(gbuffer0Desc, "GBuffer0");

	RenderGraphTextureDesc gbuffer1Desc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::RenderTarget,
		clearColor);
	gbufferData.gbuffer1 = renderGraph->CreateTexture(gbuffer1Desc, "GBuffer1");

	RenderGraphTextureDesc gbuffer2Desc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::RenderTarget,
		clearColor);
	gbufferData.gbuffer2 = renderGraph->CreateTexture(gbuffer2Desc, "GBuffer2");

	RenderGraphTextureDesc gbuffer3Desc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::RenderTarget,
		clearColor);
	gbufferData.gbuffer3 = renderGraph->CreateTexture(gbuffer3Desc, "GBuffer3");

	RenderGraphTextureDesc gbuffer4Desc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::RenderTarget,
		clearColor);
	gbufferData.gbuffer4 = renderGraph->CreateTexture(gbuffer4Desc, "GBuffer4");

	RenderGraphTextureDesc velocityBufferDesc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::RenderTarget,
		clearColor);
	gbufferData.velocityBuffer = renderGraph->CreateTexture(velocityBufferDesc, "VelocityBuffer");

	RenderGraphTextureDesc depthBufferDesc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::D32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::DepthStencil,
		clearDepth);
	depthBufferData.depthBuffer = renderGraph->CreateTexture(depthBufferDesc, "DepthBuffer");

	RenderGraphTextureDesc sceneColorDesc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::RGBA16Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess | TextureCreateFlags::RenderTarget);
	sceneColorData.sceneColor = renderGraph->CreateTexture(sceneColorDesc, "SceneColor");

	RenderGraphTextureDesc ldrTextureDesc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.targetResolutionWidth,
		perFrameData.data.targetResolutionHeight,
		PixelFormat::RGBA8Unorm,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess);
	auto ldrTexture = renderGraph->CreateTexture(ldrTextureDesc, "LDRBuffer");

	RenderGraphTextureDesc finalTextureDesc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::BGRA8Unorm,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::RenderTarget | TextureCreateFlags::UnorderedAccess);
	finalTextureData.finalTexture = renderGraph->CreateTexture(finalTextureDesc, "FinalTexture");

	RenderGraphTextureHandle historyBuffer = renderGraph->ImportExternalTexture(historyBufferCache.texture, historyBufferCache.desc, historyBufferCache.initialState, "HistoryBuffer");

	if (renderBRDFLut)
	{
		renderGraph->AddPass("BRDFLutPass", RenderGraphPassFlags::Compute | RenderGraphPassFlags::NeverGetCulled,
		[&](RenderGraphBuilder& builder)
		{
			return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
			{
				uint32 dispatchWidth = CEIL_DIV(brdfLutSize, 8);
				uint32 dispatchHeight = CEIL_DIV(brdfLutSize, 8);

				ShaderArguments shaderArguments = {};
				shaderArguments.BindTextureUAV(0, RenderBackendTextureUAVDesc::Create(brdfLut, 0));

				RenderBackendBarrier transitionBefore = RenderBackendBarrier(brdfLut, RenderBackendTextureSubresourceRange(0, 1, 0, 1), RenderBackendResourceState::Undefined, RenderBackendResourceState::UnorderedAccess);
				commandList.Transitions(&transitionBefore, 1);
				commandList.Dispatch2D(
					brdfLutShader,
					shaderArguments,
					dispatchWidth,
					dispatchHeight);
				RenderBackendBarrier transitionAfter = RenderBackendBarrier(brdfLut, RenderBackendTextureSubresourceRange(0, 1, 0, 1), RenderBackendResourceState::UnorderedAccess, RenderBackendResourceState::ShaderResource);
				commandList.Transitions(&transitionAfter, 1);
			};
		});
		renderBRDFLut = false;
	}

	renderGraph->AddPass("GBufferPass", RenderGraphPassFlags::Raster,
	[&](RenderGraphBuilder& builder)
	{
		const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
		auto& gbufferData = blackboard.Get<RenderGraphGBuffer>();
		auto& depthBufferData = blackboard.Get<RenderGraphDepthBuffer>();
		
		auto gbuffer0 = gbufferData.gbuffer0 = builder.WriteTexture(gbufferData.gbuffer0, RenderBackendResourceState::RenderTarget);
		auto gbuffer1 = gbufferData.gbuffer1 = builder.WriteTexture(gbufferData.gbuffer1, RenderBackendResourceState::RenderTarget);
		auto gbuffer2 = gbufferData.gbuffer2 = builder.WriteTexture(gbufferData.gbuffer2, RenderBackendResourceState::RenderTarget);
		auto gbuffer3 = gbufferData.gbuffer3 = builder.WriteTexture(gbufferData.gbuffer3, RenderBackendResourceState::RenderTarget);
		auto gbuffer4 = gbufferData.gbuffer4 = builder.WriteTexture(gbufferData.gbuffer4, RenderBackendResourceState::RenderTarget);
		auto velocityBuffer = gbufferData.velocityBuffer = builder.WriteTexture(gbufferData.velocityBuffer, RenderBackendResourceState::RenderTarget);
		auto depthBuffer = depthBufferData.depthBuffer = builder.WriteTexture(depthBufferData.depthBuffer, RenderBackendResourceState::DepthStencil);

		builder.BindColorTarget(0, gbuffer0, RenderTargetLoadOp::Clear, RenderTargetStoreOp::Store);
		builder.BindColorTarget(1, gbuffer1, RenderTargetLoadOp::Clear, RenderTargetStoreOp::Store);
		builder.BindColorTarget(2, gbuffer2, RenderTargetLoadOp::Clear, RenderTargetStoreOp::Store);
		builder.BindColorTarget(3, gbuffer3, RenderTargetLoadOp::Clear, RenderTargetStoreOp::Store);
		builder.BindColorTarget(4, gbuffer4, RenderTargetLoadOp::Clear, RenderTargetStoreOp::Store);
		builder.BindColorTarget(5, velocityBuffer, RenderTargetLoadOp::Clear, RenderTargetStoreOp::Store);
		builder.BindDepthTarget(depthBuffer, RenderTargetLoadOp::Clear, RenderTargetStoreOp::Store);

		return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
		{
			RenderBackendViewport viewport(0.0f, 0.0f, (float)perFrameData.data.renderResolutionWidth, (float)perFrameData.data.renderResolutionHeight);
			commandList.SetViewports(&viewport, 1);

			RenderBackendScissor scissor(0, 0, perFrameData.data.renderResolutionWidth, perFrameData.data.renderResolutionHeight);
			commandList.SetScissors(&scissor, 1);

			for (const auto& renderable : view->scene->renderables)
			{
				RenderBackendBufferHandle vertexPosition = view->scene->vertexBuffers[0][renderable.vertexBufferIndex];
				RenderBackendBufferHandle vertexNormal = view->scene->vertexBuffers[1][renderable.vertexBufferIndex];
				RenderBackendBufferHandle vertexTangent = view->scene->vertexBuffers[2][renderable.vertexBufferIndex];
				RenderBackendBufferHandle vertexTexCoord = view->scene->vertexBuffers[3][renderable.vertexBufferIndex];
				RenderBackendBufferHandle indexBuffer = view->scene->indexBuffers[renderable.indexBufferIndex];
				RenderBackendBufferHandle worldMatrixBuffer = view->scene->worldMatrixBuffer;
				RenderBackendBufferHandle prevWorldMatrixBuffer = view->scene->prevWorldMatrixBuffer;
				RenderBackendBufferHandle materialBuffer = view->scene->materialBuffer;

				ShaderArguments shaderArguments = {};
				shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
				shaderArguments.BindBuffer(1, vertexPosition, 0);
				shaderArguments.BindBuffer(2, vertexNormal, 0);
				shaderArguments.BindBuffer(3, vertexTangent, 0);
				shaderArguments.BindBuffer(4, vertexTexCoord, 0);
				shaderArguments.BindBuffer(5, worldMatrixBuffer, renderable.transformIndex);
				shaderArguments.BindBuffer(6, prevWorldMatrixBuffer, renderable.transformIndex);
				shaderArguments.BindBuffer(7, materialBuffer, renderable.materialIndex);
				//shaderArguments.BindBuffer(8, indirectBuffer, 0);

				commandList.DrawIndexed(
					gbufferShader,
					shaderArguments,
					indexBuffer,
					renderable.numIndices,
					1,
					renderable.firstIndex,
					0,
					0,
					PrimitiveTopology::TriangleList);
			}
		};
	});

	RenderGraphTextureDesc motionVectorsDesc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::RGBA32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess);

	auto motionVectors = renderGraph->CreateTexture(motionVectorsDesc, "MotionVectors");
	auto motionVectors1 = renderGraph->CreateTexture(motionVectorsDesc, "MotionVectors1");

	renderGraph->AddPass("MotionVectorsPass", RenderGraphPassFlags::Compute,
	[&](RenderGraphBuilder& builder)
	{
		const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
		const auto& depthBufferData = blackboard.Get<RenderGraphDepthBuffer>();

		auto depthBuffer = builder.ReadTexture(depthBufferData.depthBuffer, RenderBackendResourceState::ShaderResource);
		motionVectors = builder.WriteTexture(motionVectors, RenderBackendResourceState::UnorderedAccess);
		motionVectors1 = builder.WriteTexture(motionVectors1, RenderBackendResourceState::UnorderedAccess);
		
		return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
		{
			uint32 dispatchWidth = CEIL_DIV(perFrameData.data.renderResolutionWidth, 8);
			uint32 dispatchHeight = CEIL_DIV(perFrameData.data.renderResolutionHeight, 8);

			ShaderArguments shaderArguments = {};
			shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
			shaderArguments.BindTextureSRV(1, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(depthBuffer)));
			shaderArguments.BindTextureUAV(2, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(motionVectors), 0));
			shaderArguments.BindTextureUAV(3, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(motionVectors1), 0));

			commandList.Dispatch2D(
				motionVectorsShader,
				shaderArguments,
				dispatchWidth,
				dispatchHeight);
		};
	});

	RenderGraphTextureDesc shadowMaskDesc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::R32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess);
	auto shadowMask = renderGraph->CreateTexture(shadowMaskDesc, "ShadowMask");

	if (mainLight.UseRayTracingShadows())
	{
		RenderRayTracingShadows(*renderGraph, blackboard, *view, mainLight, shadowMapParameters, shadowMask);
	}
	else // Cascaded Shadow Mapping
	{
		RenderScreenSpaceShadows(*renderGraph, blackboard, *view, mainLight, shadowMapParameters, shadowMask);

		if (true)
		{
			auto filteredShadowMask = renderGraph->CreateTexture(shadowMaskDesc, "FilteredShadowMask");

			DenoiseShadowMaskSSD(*renderGraph, blackboard, *view, filteredShadowMask, shadowMask);

			// DenoiseShadowMaskSVGF();
		}
	}
	
	// GTAO
	// AddGTAOPasses();

	RenderGraphTextureDesc horizonSearchIntergralOutputTextureDesc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.renderResolutionWidth,
		perFrameData.data.renderResolutionHeight,
		PixelFormat::R32Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess);
	RenderGraphTextureHandle horizonSearchIntergralOutputTexture = renderGraph->CreateTexture(horizonSearchIntergralOutputTextureDesc, "GTAOHorizonSearchIntergralOutputTexture");

	renderGraph->AddPass("GTAOHorizonSearchIntegralPass", RenderGraphPassFlags::Compute,
	[&](RenderGraphBuilder& builder)
	{
		const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
		const auto& gbufferData = blackboard.Get<RenderGraphGBuffer>();
		const auto& depthBufferData = blackboard.Get<RenderGraphDepthBuffer>();

		auto gbuffer1 = builder.ReadTexture(gbufferData.gbuffer1, RenderBackendResourceState::ShaderResource);
		auto gbuffer2 = builder.ReadTexture(gbufferData.gbuffer2, RenderBackendResourceState::ShaderResource);
		auto depthBuffer = builder.ReadTexture(depthBufferData.depthBuffer, RenderBackendResourceState::ShaderResource);
		
		horizonSearchIntergralOutputTexture = builder.WriteTexture(horizonSearchIntergralOutputTexture, RenderBackendResourceState::UnorderedAccess);

		return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
		{
			uint32 dispatchWidth = CEIL_DIV(perFrameData.data.renderResolutionWidth, 8);
			uint32 dispatchHeight = CEIL_DIV(perFrameData.data.renderResolutionHeight, 8);

			ShaderArguments shaderArguments = {};
			shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
			shaderArguments.BindTextureSRV(1, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(depthBuffer)));
			shaderArguments.BindTextureSRV(2, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(gbuffer1)));
			shaderArguments.BindTextureSRV(3, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(gbuffer2)));
			shaderArguments.BindTextureUAV(4, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(horizonSearchIntergralOutputTexture), 0));

			RenderBackendShaderHandle computeShader = gtaoMainShader;
			commandList.Dispatch2D(
				computeShader,
				shaderArguments,
				dispatchWidth,
				dispatchHeight);
		};
	});

	renderGraph->AddPass("LightingPass", RenderGraphPassFlags::Compute,
	[&](RenderGraphBuilder& builder)
	{
		const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
		const auto& gbufferData = blackboard.Get<RenderGraphGBuffer>();
		auto& sceneColorData = blackboard.Get<RenderGraphSceneColor>();
		auto gbuffer0 = builder.ReadTexture(gbufferData.gbuffer0, RenderBackendResourceState::ShaderResource);
		auto gbuffer1 = builder.ReadTexture(gbufferData.gbuffer1, RenderBackendResourceState::ShaderResource);
		auto gbuffer2 = builder.ReadTexture(gbufferData.gbuffer2, RenderBackendResourceState::ShaderResource);
		auto gbuffer3 = builder.ReadTexture(gbufferData.gbuffer3, RenderBackendResourceState::ShaderResource);
		horizonSearchIntergralOutputTexture = builder.ReadTexture(horizonSearchIntergralOutputTexture, RenderBackendResourceState::ShaderResource);
		
#if DEBUG_ONLY_RAY_TRACING_ENBALE
		svgfFilteredIllumination = builder.ReadTexture(svgfFilteredIllumination, RenderBackendResourceState::ShaderResource);
#else
		shadowMask = builder.ReadTexture(shadowMask, RenderBackendResourceState::ShaderResource);
#endif

		auto sceneColor = sceneColorData.sceneColor = builder.WriteTexture(sceneColorData.sceneColor, RenderBackendResourceState::UnorderedAccess);

		return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
		{
			uint32 dispatchWidth = CEIL_DIV(perFrameData.data.renderResolutionWidth, 8);
			uint32 dispatchHeight = CEIL_DIV(perFrameData.data.renderResolutionHeight, 8);

			ShaderArguments shaderArguments = {};
			shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
			shaderArguments.BindTextureSRV(1, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(gbuffer0)));
			shaderArguments.BindTextureSRV(2, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(gbuffer1)));
			shaderArguments.BindTextureSRV(3, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(gbuffer2)));
			shaderArguments.BindTextureSRV(4, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(gbuffer3)));
			shaderArguments.BindTextureSRV(5, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(horizonSearchIntergralOutputTexture)));
#if DEBUG_ONLY_RAY_TRACING_ENBALE
			shaderArguments.BindTextureSRV(6, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(svgfFilteredIllumination)));
#else
			shaderArguments.BindTextureSRV(6, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(shadowMask)));
#endif
			shaderArguments.BindTextureUAV(7, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(sceneColor), 0));
			shaderArguments.BindTextureSRV(8, RenderBackendTextureSRVDesc::Create(brdfLut));
			shaderArguments.BindTextureSRV(9, RenderBackendTextureSRVDesc::Create(view->scene->GetSkyLight().GetIrradianceEnvironmentMap()));
			shaderArguments.BindTextureSRV(10, RenderBackendTextureSRVDesc::Create(view->scene->GetSkyLight().GetFilteredEnvironmentMap()));

			commandList.Dispatch2D(
				lightingShader,
				shaderArguments,
				dispatchWidth,
				dispatchHeight);
		};
	});

	//const bool renderSkyAtmosphere = ShouldRenderSkyAtmosphere();
	//const bool renderSkyAtmosphere = false;
	//if (renderSkyAtmosphere)
	//{
	//	const auto& entities = view->scene->GetEntityManager()->GetView<SkyAtmosphereComponent>();
	//	for (const auto& entity : entities)
	//	{
	//		RenderSky(*renderGraph, *skyAtmosphere, view->scene->GetEntityManager()->GetComponent<SkyAtmosphereComponent>(entity));
	//		break;
	//	}
	//}

	RenderGraphTextureDesc dofTextureDesc = RenderGraphTextureDesc::Create2D(
		perFrameData.data.targetResolutionWidth,
		perFrameData.data.targetResolutionHeight,
		PixelFormat::RGBA16Float,
		TextureCreateFlags::ShaderResource | TextureCreateFlags::UnorderedAccess);
	auto dofTexture = renderGraph->CreateTexture(dofTextureDesc, "DOFBuffer");

	//renderGraph->AddPass("Depth of Field Pass", RenderGraphPassFlags::Compute,
	//[&](RenderGraphBuilder& builder)
	//{
	//	const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
	//	const auto& sceneColorData = blackboard.Get<RenderGraphSceneColor>();

	//	auto depthBuffer = builder.ReadTexture(depthBufferData.depthBuffer, RenderBackendResourceState::ShaderResource);
	//	auto sceneColor = builder.ReadTexture(sceneColorData.sceneColor, RenderBackendResourceState::ShaderResource);
	//	dofTexture = builder.WriteTexture(dofTexture, RenderBackendResourceState::UnorderedAccess);

	//	return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
	//	{
	//		uint32 dispatchWidth = CEIL_DIV(perFrameData.data.targetResolutionWidth, POST_PROCESS_THREAD_GROUP_SIZE);
	//		uint32 dispatchHeight = CEIL_DIV(perFrameData.data.targetResolutionHeight, POST_PROCESS_THREAD_GROUP_SIZE);

	//		ShaderArguments shaderArguments = {};
	//		shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
	//		shaderArguments.BindTextureSRV(1, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(depthBuffer)));
	//		shaderArguments.BindTextureSRV(2, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(sceneColor)));
	//		shaderArguments.BindTextureUAV(3, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(dofTexture), 0));

	//		commandList.Dispatch2D(
	//			dofShader,
	//			shaderArguments,
	//			dispatchWidth,
	//			dispatchHeight);
	//	};
	//});

	renderGraph->AddPass("SkyBoxPass", RenderGraphPassFlags::Raster,
		[&](RenderGraphBuilder& builder)
		{
			const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
			auto& depthBufferData = blackboard.Get<RenderGraphDepthBuffer>();

			auto depthBuffer = builder.ReadTexture(depthBufferData.depthBuffer, RenderBackendResourceState::DepthStencil);
			auto sceneColor = sceneColorData.sceneColor = builder.ReadWriteTexture(sceneColorData.sceneColor, RenderBackendResourceState::RenderTarget);

			builder.BindColorTarget(0, sceneColor, RenderTargetLoadOp::Load, RenderTargetStoreOp::Store);
			builder.BindDepthTarget(depthBufferData.depthBuffer, RenderTargetLoadOp::Load, RenderTargetStoreOp::Store);

			return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
			{
				uint32 dispatchWidth = CEIL_DIV(perFrameData.data.targetResolutionWidth, POST_PROCESS_THREAD_GROUP_SIZE);
				uint32 dispatchHeight = CEIL_DIV(perFrameData.data.targetResolutionHeight, POST_PROCESS_THREAD_GROUP_SIZE);

				ShaderArguments shaderArguments = {};
				shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
				shaderArguments.BindTextureSRV(1, RenderBackendTextureSRVDesc::Create(view->scene->GetSkyLight().GetEnvironmentMap()));
				shaderArguments.PushConstants(0, 0.0f);

				commandList.Draw(
					skyBoxShader,
					shaderArguments,
					3, 1, 0, 0,
					PrimitiveTopology::TriangleList);
			};
		});

	renderGraph->AddPass("TonemappingPass", RenderGraphPassFlags::Compute,
	[&](RenderGraphBuilder& builder)
	{
		const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();

		auto sceneColor = builder.ReadTexture(sceneColorData.sceneColor, RenderBackendResourceState::ShaderResource);
		ldrTexture = builder.WriteTexture(ldrTexture, RenderBackendResourceState::UnorderedAccess);

		return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
		{
			uint32 dispatchWidth = CEIL_DIV(perFrameData.data.targetResolutionWidth, POST_PROCESS_THREAD_GROUP_SIZE);
			uint32 dispatchHeight = CEIL_DIV(perFrameData.data.targetResolutionHeight, POST_PROCESS_THREAD_GROUP_SIZE);

			ShaderArguments shaderArguments = {};
			shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
			shaderArguments.BindTextureSRV(1, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(sceneColor)));
			shaderArguments.BindTextureUAV(2, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(ldrTexture), 0));

			commandList.Dispatch2D(
				tonemappingShader,
				shaderArguments,
				dispatchWidth,
				dispatchHeight);
		};
	});

	renderGraph->AddPass("FXAAPass", RenderGraphPassFlags::Compute,
	[&](RenderGraphBuilder& builder)
	{
		const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();
		const auto& sceneColorData = blackboard.Get<RenderGraphSceneColor>();
		auto& finalTextureData = blackboard.Get<RenderGraphFinalTexture>();

		ldrTexture = builder.ReadTexture(ldrTexture, RenderBackendResourceState::ShaderResource);
		auto finalTexture = finalTextureData.finalTexture = builder.WriteTexture(finalTextureData.finalTexture, RenderBackendResourceState::UnorderedAccess);

		return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
		{
			uint32 dispatchWidth = CEIL_DIV(perFrameData.data.targetResolutionWidth, POST_PROCESS_THREAD_GROUP_SIZE);
			uint32 dispatchHeight = CEIL_DIV(perFrameData.data.targetResolutionHeight, POST_PROCESS_THREAD_GROUP_SIZE);

			ShaderArguments shaderArguments = {};
			shaderArguments.BindBuffer(0, perFrameDataBuffer, 0);
			shaderArguments.BindTextureSRV(1, RenderBackendTextureSRVDesc::Create(registry.GetRenderBackendTexture(ldrTexture)));
			shaderArguments.BindTextureUAV(2, RenderBackendTextureUAVDesc::Create(registry.GetRenderBackendTexture(finalTexture), 0));

			commandList.Dispatch2D(
				fxaaShader,
				shaderArguments,
				dispatchWidth,
				dispatchHeight);
		};
	});

	renderGraph->ExportTextureDeferred(ldrTexture, &historyBufferCache);

	if (false)
	{
	    auto captureTarget0 = renderGraph->ImportExternalTexture(view->captureTargets[0], view->captureTargetDescs[0], RenderBackendResourceState::Undefined, "CaptureTarget0");
		renderGraph->AddPass("CapturePass", RenderGraphPassFlags::NeverGetCulled,
		[&](RenderGraphBuilder& builder)
		{
			ldrTexture = builder.ReadTexture(ldrTexture, RenderBackendResourceState::CopySrc);
			captureTarget0 = builder.WriteTexture(captureTarget0, RenderBackendResourceState::CopyDst);

			return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
			{
				uint32 width = perFrameData.data.targetResolutionWidth;
				uint32 height = perFrameData.data.targetResolutionHeight;

				commandList.CopyTexture2D(
					registry.GetRenderBackendTexture(ldrTexture),
					{ 0, 0 }, 
					0,
					registry.GetRenderBackendTexture(captureTarget0),
					{ 0, 0 }, 
					0,
					{ width, height });
			};
		});
	}

#if HE_EDITOR
	RenderGizmo();
#endif

	renderGraph->AddPass("ImGuiPass", RenderGraphPassFlags::Raster | RenderGraphPassFlags::SkipRenderPass,
	[&](RenderGraphBuilder& builder)
	{
		auto& finalTextureData = blackboard.Get<RenderGraphFinalTexture>();
		auto finalTexture = finalTextureData.finalTexture = builder.ReadWriteTexture(finalTextureData.finalTexture, RenderBackendResourceState::RenderTarget);

		builder.BindColorTarget(0, finalTexture, RenderTargetLoadOp::Load, RenderTargetStoreOp::Store);

		return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
		{
			uiRenderer->Render(commandList, registry.GetRenderBackendTexture(finalTexture));
		};
	});

	CooyRenderGraphFinalTextureToCameraTarget(renderGraph);

	// Export resources
}

}