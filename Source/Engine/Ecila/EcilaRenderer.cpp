#pragma once

#include "DaisyRenderer.h"

namespace HE
{

	struct PerFrameData
	{
		uint32 frameIndex = 0;

		float gamma;
		float exposure;

		Vector3 sunDirection;
		Vector3 solarIrradiance;
		float solarAngularRadius;
		Vector3 sunIlluminanceScale;

		Vector3 cameraPosition;
		float nearPlane;
		float farPlane;

		Matrix4x4 viewMatrix;
		Matrix4x4 invViewMatrix;
		Matrix4x4 projectionMatrix;
		Matrix4x4 invProjectionMatrix;
		Matrix4x4 viewProjectionMatrix;
		Matrix4x4 invViewProjectionMatrix;
		Matrix4x4 prevViewProjectionMatrix;

		uint32 renderResolutionWidth;
		uint32 renderResolutionHeight;
		uint32 targetResolutionWidth;
		uint32 targetResolutionHeight;
	};

	struct ShadowMapShaderParameters
	{
		Matrix4x4 viewProjectionMatrix[4];
		float splitDepth[4];
		float shadowMapInvSize;
	};

	struct RenderGraphPerFrameData
	{
		PerFrameData data;
		RenderBackendBufferHandle buffer;
	};
	RENDER_GRAPH_BLACKBOARD_REGISTER_STRUCT(RenderGraphPerFrameData);

	struct RenderGraphGBuffer
	{
		RenderGraphTextureHandle gbuffer0;
		RenderGraphTextureHandle gbuffer1;
		RenderGraphTextureHandle gbuffer2;
		RenderGraphTextureHandle gbuffer3;
		RenderGraphTextureHandle gbuffer4;
		RenderGraphTextureHandle velocityBuffer;
	};
	RENDER_GRAPH_BLACKBOARD_REGISTER_STRUCT(RenderGraphGBuffer);

	struct RenderGraphDepthBuffer
	{
		RenderGraphTextureHandle depthBuffer;
	};
	RENDER_GRAPH_BLACKBOARD_REGISTER_STRUCT(RenderGraphDepthBuffer);

	struct RenderGraphSceneColor
	{
		RenderGraphTextureHandle sceneColor;
	};
	RENDER_GRAPH_BLACKBOARD_REGISTER_STRUCT(RenderGraphSceneColor);

	struct RenderGraphFinalTexture
	{
		RenderGraphTextureHandle finalTexture;
	};
	RENDER_GRAPH_BLACKBOARD_REGISTER_STRUCT(RenderGraphFinalTexture);

	struct RenderGraphOutputTexture
	{
		RenderBackendTextureDesc outputTextureDesc;
		RenderGraphTextureHandle outputTexture;
	};
	RENDER_GRAPH_BLACKBOARD_REGISTER_STRUCT(RenderGraphOutputTexture);

	void DaisyRenderer::RenderScreenSpaceShadows(
		RenderGraph& renderGraph,
		RenderGraphBlackboard& blackboard,
		const SceneView& view,
		const LightRenderProxy& lightProxy, 
		RenderBackendBufferHandle parametersBuffer, 
		RenderGraphTextureHandle& shadowMask)
	{
		uint32 numDynamicShadowCascades = lightProxy.numDynamicShadowCascades;
		uint32 shadowMapSize = 4096;

		RenderGraphTextureDesc shadowMapDesc = RenderGraphTextureDesc::Create2DArray(
			shadowMapSize,
			shadowMapSize,
			PixelFormat::D32Float,
			TextureCreateFlags::ShaderResource | TextureCreateFlags::DepthStencil,
			numDynamicShadowCascades,
			RenderTargetClearValue::CreateDepthValue(0.0f));

		auto shadowMap = renderGraph->CreateTexture(shadowMapDesc, "ShadowMap");

		ShadowMapShaderParameters parameters = {};
		for (uint32 cascadeIndex = 0; cascadeIndex < numDynamicShadowCascades; cascadeIndex++)
		{
			parameters.viewProjectionMatrix[cascadeIndex] = lightProxy.viewProjectionMatrix[cascadeIndex];
			parameters.splitDepth[cascadeIndex] = lightProxy.splitDepth[cascadeIndex];
			parameters.shadowMapInvSize = 1.0f / shadowMapSize;
		}

		void* data = nullptr;
		RenderBackendMapBuffer(renderBackend, parametersBuffer, &data);
		memcpy((uint8*)data, &parameters, sizeof(ShadowMapShaderParameters));
		RenderBackendUnmapBuffer(renderBackend, parametersBuffer);
	 
		for (uint32 cascadeIndex = 0; cascadeIndex < numDynamicShadowCascades; cascadeIndex++)
		{
			renderGraph->AddPass("ShadowMapPass", RenderGraphPassFlags::Raster,
				[&](RenderGraphBuilder& builder)
				{
					const auto& perFrameData = blackboard.Get<RenderGraphPerFrameData>();

					shadowMap = builder.WriteTexture(shadowMap, RenderBackendResourceState::DepthStencil);

					builder.BindDepthTarget(shadowMap, RenderTargetLoadOp::Clear, RenderTargetStoreOp::Store, 0, cascadeIndex);

					return [=](RenderGraphRegistry& registry, RenderCommandList& commandList)
					{
						RenderBackendViewport viewport(0.0f, 0.0f, (float)shadowMapSize, (float)shadowMapSize);
						RenderBackendScissor scissor(0, 0, shadowMapSize, shadowMapSize);

						commandList.SetViewports(&viewport, 1);
						commandList.SetScissors(&scissor, 1);

						for (const auto& renderable : view->scene->renderables)
						{
							RenderBackendBufferHandle vertexPosition = view->scene->vertexBuffers[0][renderable.vertexBufferIndex];
							RenderBackendBufferHandle indexBuffer = view->scene->indexBuffers[renderable.indexBufferIndex];
							RenderBackendBufferHandle worldMatrixBuffer = view->scene->worldMatrixBuffer;

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
		}
	}


	{
		lightProxy->UpdateShadowCascades(view->camera);

		RenderGraphTextureDesc shadowMapDesc = RenderGraphTextureDesc::Create2DArray(
			shadowMapSize,
			shadowMapSize,
			PixelFormat::D32Float,
			TextureCreateFlags::ShaderResource | TextureCreateFlags::DepthStencil,
			numDynamicShadowCascades,
			clearDepth);

		auto shadowMap = renderGraph->CreateTexture(shadowMapDesc, "ShadowMap");
	}
}