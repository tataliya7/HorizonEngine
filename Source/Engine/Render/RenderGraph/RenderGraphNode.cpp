module HorizonEngine.Render.RenderGraph:RenderGraphNode;

namespace HE
{
	void RenderGraphDAG::RegisterNode(RenderGraphNode* node)
	{
		nodes.push_back(node);
	}

	void RenderGraphDAG::Clear()
	{
		nodes.clear();
	}
}