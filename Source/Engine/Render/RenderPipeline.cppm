module;

export module HorizonEngine.Render.RenderPipeline;

import HorizonEngine.Core;
import HorizonEngine.Render.Core;

export namespace HE
{
	class RenderGraph;

	class RenderPipeline
	{
	public:
		virtual void SetupRenderGraph(SceneView* view, RenderGraph* renderGraph) = 0;
	};
}