#pragma once

#include "ECS/ECSCommon.h"
#include "ECS/Entity.h"

namespace HE
{
	struct SceneHierarchyComponent
	{
		uint32 depth;
		uint32 numChildren;
		EntityHandle parent;
		EntityHandle first;
		EntityHandle next;
		EntityHandle prev;

		SceneHierarchyComponent()
			: depth(0) 
			, numChildren(0)
			, parent()
			, first()
			, next()
			, prev() 
		{
			using namespace entt;
			auto factory = entt::meta<SceneHierarchyComponent>();
			factory.data<&SceneHierarchyComponent::prev, entt::as_ref_t>("Prev"_hs)
				.prop("Name"_hs, std::string("Prev"));
			factory.data<&SceneHierarchyComponent::next, entt::as_ref_t>("Next"_hs)
				.prop("Name"_hs, std::string("Next"));
			factory.data<&SceneHierarchyComponent::first, entt::as_ref_t>("First"_hs)
				.prop("Name"_hs, std::string("First"));
			factory.data<&SceneHierarchyComponent::parent, entt::as_ref_t>("Parent"_hs)
				.prop("Name"_hs, std::string("Parent"));
			factory.data<&SceneHierarchyComponent::numChildren, entt::as_ref_t>("Num Children"_hs)
				.prop("Name"_hs, std::string("Num Children"));
			factory.data<&SceneHierarchyComponent::depth, entt::as_ref_t>("Depth"_hs)
				.prop("Name"_hs, std::string("Depth"));
		}
	};
}
