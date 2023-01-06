module;

#include <sstream>

module HorizonEngine.Render.RenderGraph;

namespace HE
{
	void RenderGraphPass::Graphviz(std::stringstream& stream) const
	{
		stream << "\t\t" << name << " [color=orange];\n";

		for (RenderGraphNode* input : inputs)
		{
			stream << "\t\t" << input->GetName() << " [color=green];\n";
			stream << "\t\t" << input->GetName() << " -> " << name << "\n";
		}

		for (RenderGraphNode* output : outputs)
		{
			stream << "\t\t" << output->GetName() << " [color=green];\n";
			stream << "\t\t" << name << " -> " << output->GetName() << "\n";
		}
	}
}