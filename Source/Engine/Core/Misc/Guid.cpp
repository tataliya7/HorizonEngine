module;

#include "Core/CoreCommon.h"

#include <windows.h>

module HorizonEngine.Core.Misc;

namespace HE
{
#if defined(HE_PLATFORM_WINDOWS)
#include <objbase.h>
	void GenerateGuidImpl(Guid* guid)
	{
		ASSERT(CoCreateGuid((GUID*)guid) == S_OK);
	}
#endif

	Guid Guid::Generate()
	{
		Guid guid;
		GenerateGuidImpl(&guid);
		return guid;
	}

	std::string Guid::ToString(const Guid& guid)
	{
		return std::string();
	}
}