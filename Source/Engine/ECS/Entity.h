#pragma once

#include "ECS/ECSCommon.h"

namespace HE
{
	class EntityHandle
	{
	public:
		EntityHandle() : handle(entt::null) {}
		EntityHandle(uint64 entity) : handle((entt::entity)entity) {}
		EntityHandle(const entt::entity& entity) : handle(entity) {}
		EntityHandle(const EntityHandle& other) : handle(other.handle) {}
		FORCEINLINE bool operator==(const EntityHandle& rhs) const
		{
			return handle == rhs.handle;
		}
		FORCEINLINE bool operator!=(const EntityHandle& rhs) const
		{
			return handle != rhs.handle;
		}
		FORCEINLINE operator uint64() const 
		{ 
			return (uint64)handle;
		}
		FORCEINLINE operator bool() const
		{
			return handle == entt::null;
		}
		FORCEINLINE operator entt::entity() const
		{
			return handle;
		}
	private:
		entt::entity handle;
	};

	class EntityManager
	{
	public:

		EntityManager() {}
		~EntityManager() {}

		void Clear();

		entt::registry* Get()
		{
			return &registry;
		}

		EntityHandle CreateEntity(const char* name);

		void DestroyEntity(EntityHandle entity);

		bool HasEntity(EntityHandle entity);

		template<typename... Components>
		auto GetGroup()
		{
			return registry.group<Components ...>();
		}

		template<typename... Components>
		auto GetGroup() const
		{ 
			return registry.group<Components ...>(); 
		}

		template<typename... Components>
		auto GetView()
		{
			return registry.view<Components ...>();
		}

		template<typename... Components>
		auto GetView() const
		{ 
			return registry.view<Components ...>(); 
		}

		template<typename Component>
		auto OnConstruct() 
		{ 
			return registry.on_construct<Component>();
		}

		template<typename Component>
		auto OnUpdate() 
		{ 
			return registry.on_update<Component>();
		}

		template<typename Component>
		bool HasComponent(EntityHandle entity)
		{
			return registry.all_of<Component>(entity);
		}

		template<typename... Component>
		bool HasAllComponents(EntityHandle entity) 
		{ 
			return registry.all_of<Component ...>(entity);
		}

		template<typename... Component>
		bool HasAnyComponent(EntityHandle entity) 
		{ 
			return registry.any_of<Component ...>(entity);
		}

		template<typename Component, typename... Args>
		Component& AddComponent(EntityHandle entity, Args&&... args)
		{
			return registry.emplace<Component>(entity, std::forward<Args>(args)...);
		}

		template<typename Component, typename... Args>
		void AddOrReplaceComponent(EntityHandle entity, Args&&... args) 
		{ 
			registry.emplace_or_replace<Component>(entity, std::forward<Args>(args)...);
		}

		template<typename Component, typename... Args>
		void ReplaceComponent(EntityHandle entity, Args&&... args) 
		{
			registry.replace<Component>(entity, std::forward<Args>(args)...);
		}

		template<typename Component>
		Component& GetComponent(EntityHandle entity) const 
		{ 
			return registry.get<Component>(entity);
		}

		template<typename Component>
		Component& GetComponent(EntityHandle entity)
		{
			return registry.get<Component>(entity);
		}

		template<typename... Component>
		decltype(auto) GetComponents(EntityHandle entity) const
		{ 
			return registry.get<Component ...>(entity);
		}

		template<typename... Component>
		decltype(auto) GetComponents(EntityHandle entity)
		{
			return registry.get<Component ...>(entity);
		}

		template<typename Component, typename... Args>
		Component& GetOrAddComponent(EntityHandle entity, Args&&... args)
		{
			return registry.get_or_emplace<Component>(entity, std::forward<Args>(args)...);
		}

		template<typename Component>
		Component* TryGetComponent(EntityHandle entity) 
		{ 
			return registry.try_get<Component>(entity);
		}

		template<typename... Component>
		auto TryGetComponents(EntityHandle entity) 
		{ 
			return registry.try_get<Component ...>(entity);
		}

		template<typename Component>
		auto RemoveComponent(EntityHandle entity) 
		{ 
			return registry.remove<Component>(entity);
		}

	private:
		entt::registry registry;
	};
}
