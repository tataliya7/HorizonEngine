#pragma once

#include "ECS/ECSCommon.h"

namespace HE
{
	//struct GuidComponent
	//{
	//	Guid guid;
	//	inline std::string ToString() const
	//	{
	//		return Guid::ToString(guid);
	//	}
	//};

	struct NameComponent
	{
		std::string name;
		NameComponent() = default;
		NameComponent(const NameComponent& other) = default;
		NameComponent(const std::string& name) : name(name) {}
		inline operator std::string& () { return name; }
		inline operator const std::string& () const { return name; }
		inline void operator=(const std::string& str) { name = str; }
		inline bool operator==(const std::string& str) const { return name.compare(str) == 0; }
	};
}
