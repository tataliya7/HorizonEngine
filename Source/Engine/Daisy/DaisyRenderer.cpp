#include "DaisyRenderer.h"
#include "DxcShaderCompiler/DxcShaderCompiler.h"
#include "HybridRenderPipeline/HybridRenderPipeline.h"

import HorizonEngine.Render.VulkanRenderBackend;

namespace HE
{
	DaisyRenderer::DaisyRenderer(void* window)
	{
		arena = new LinearArena(nullptr, 1048576);

		shaderCompiler = CreateDxcShaderCompiler();

		//int flags = VULKAN_RENDER_BACKEND_CREATE_FLAGS_SURFACE;
		int flags = VULKAN_RENDER_BACKEND_CREATE_FLAGS_VALIDATION_LAYERS | VULKAN_RENDER_BACKEND_CREATE_FLAGS_SURFACE;

		renderBackend = VulkanRenderBackendCreateBackend(flags);

		uint32 physicalDeviceID = 0;
		RenderBackendCreateRenderDevices(renderBackend, &physicalDeviceID, 1, &primaryDeviceMask);

		uiRenderer = new UIRenderer(window, renderBackend, shaderCompiler);
		uiRenderer->Init();

		RenderBackendTimingQueryHeapDesc tDesc(MaxNumTimingQueryRegions);

		renderContext = new RenderContext();
		renderContext->arena = arena;
		renderContext->renderBackend = renderBackend;
		renderContext->shaderCompiler = shaderCompiler;
		renderContext->uiRenderer = uiRenderer;
		renderContext->timingQueryHeap = RenderBackendCreateTimingQueryHeap(renderBackend, ~0u, &tDesc, "DefaultTimingQueryHeap");

		renderPipeline = new HybridRenderPipeline(renderContext);
		renderPipeline->Init();

		GGlobalShaderLibrary = new ShaderLibrary(renderBackend, shaderCompiler);
	}

	DaisyRenderer::~DaisyRenderer()
	{
		delete renderPipeline;
		VulkanRenderBackendDestroyBackend(renderBackend);
		DestroyDxcShaderCompiler(shaderCompiler);
		uiRenderer->Shutdown();
		delete uiRenderer;
		delete arena;
	}

	void DaisyRenderer::BeginFrame()
	{
		uiRenderer->BeginFrame();
	}

	void DaisyRenderer::EndFrame()
	{
		((LinearArena*)arena)->Reset();
	}

	static void UpdateCameraMatrices(Camera& camera)
	{
		static Quaternion zUpQuat = glm::rotate(glm::quat(), Math::DegreesToRadians(90.0), Vector3(1.0, 0.0, 0.0));
		camera.invViewMatrix = Math::Compose(camera.position, Quaternion(Math::DegreesToRadians(camera.euler)) * zUpQuat, Vector3(1.0f, 1.0f, 1.0f));
		camera.viewMatrix = Math::Inverse(camera.invViewMatrix);
		camera.projectionMatrix = Math::PerspectiveReverseZ_RH_ZO(Math::DegreesToRadians(camera.fieldOfView), camera.aspectRatio, camera.zNear, camera.zFar);
		camera.invProjectionMatrix = Math::Inverse(camera.projectionMatrix);
	}

	void DaisyRenderer::Render(SceneView* view)
	{
		view->renderPipeline = renderPipeline;
		
		UpdateCameraMatrices(view->camera);

		uiRenderer->EndFrame();
		RenderSceneView(renderContext, view);

		GRenderGraphResourcePool->Tick();
	}
}