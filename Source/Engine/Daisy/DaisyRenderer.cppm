export module HorizonEngine.Daisy;

import HorizonEngine.Core;
import HorizonEngine.Render;

export namespace HE
{
	class HybridRenderPipeline;

	class DaisyRenderer : public Renderer
	{
	public:
		// TODO: Remove window pointer
		DaisyRenderer(void* window);
		~DaisyRenderer();
		void BeginFrame();
		void EndFrame();
		virtual void Render(SceneView* view) override;
		RenderBackend* GetRenderBackend()
		{
			return renderBackend;
		}
		uint32 GetPrimaryDeviceMask() const
		{
			return primaryDeviceMask;
		}
		MemoryArena* arena = nullptr;
	private:
		ShaderCompiler* shaderCompiler = nullptr;
		RenderBackend* renderBackend = nullptr;
		ShaderLibrary* shaderLibrary = nullptr;
		RenderContext* renderContext = nullptr;
		UIRenderer* uiRenderer = nullptr;
		HybridRenderPipeline* renderPipeline = nullptr;
		uint32 primaryDeviceMask = ~0u;
	};

}