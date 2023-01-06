module;

#include <map>
#include <string>
#include <sstream>

module HorizonEngine.Render.RenderGraph:RenderGraphBlackboard;

namespace HE
{
	std::string RenderGraphBlackboard::GetStructName(const char* structName, const char* filename, uint32 line)
	{
		std::ostringstream result; 
		result << structName << filename << line;
		return result.str();
	}

	uint32 RenderGraphBlackboard::AllocateIndex(std::string&& structName)
	{
		static std::map<std::string, uint32> structIndexMap;
		static uint32 nextIndex = 0;

		uint32 result;
		const auto& found = structIndexMap.find(structName);
		if (found != structIndexMap.end())
		{
			result = found->second;
		}
		else
		{
			structIndexMap.emplace(std::move(structName), nextIndex);
			result = nextIndex;
			nextIndex++;
		}
		return result;
	}
}
