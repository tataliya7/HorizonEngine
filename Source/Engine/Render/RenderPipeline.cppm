module;

#include <vector>

export module HorizonEngine.Render.RenderPipeline;

import HorizonEngine.Core;
import HorizonEngine.Render.Core;

export namespace HE
{
	class RenderGraph;
	struct SceneView;

	class RenderPipeline
	{
	public:
		virtual void SetupRenderGraph(SceneView* view, RenderGraph* renderGraph) = 0;
	};
}