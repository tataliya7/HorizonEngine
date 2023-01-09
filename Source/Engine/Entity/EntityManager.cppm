module;

#include "Core/CoreDefinitions.h"

export module HorizonEngine.Entity.Manager;

export import "entt/entt.hpp";

import HorizonEngine.Core;

export namespace HE
{
	class EntityHandle
	{
	public:
		static EntityHandle Null;
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

		FORCEINLINE void Clear()
		{
			registry.clear();
		}

		FORCEINLINE entt::registry* Get()
		{
			return &registry;
		}

		FORCEINLINE EntityHandle CreateEntity()
		{
			EntityHandle entity(registry.create());
			return entity;
		}

		FORCEINLINE void DestroyEntity(EntityHandle entity)
		{
			registry.destroy(entity);
		}

		FORCEINLINE bool HasEntity(EntityHandle entity)
		{
			return registry.valid(entity);
		}

		template<typename... Components>
		FORCEINLINE auto GetGroup()
		{
			return registry.group<Components ...>();
		}

		template<typename... Components>
		FORCEINLINE auto GetGroup() const
		{
			return registry.group<Components ...>();
		}

		template<typename... Components>
		FORCEINLINE auto GetView()
		{
			return registry.view<Components ...>();
		}

		template<typename... Components>
		FORCEINLINE auto GetView() const
		{
			return registry.view<Components ...>();
		}

		template<typename Component>
		FORCEINLINE auto OnConstruct()
		{
			return registry.on_construct<Component>();
		}

		template<typename Component>
		FORCEINLINE auto OnDestroy()
		{
			return registry.on_destroy<Component>();
		}

		template<typename Component>
		FORCEINLINE auto OnUpdate()
		{
			return registry.on_update<Component>();
		}

		template<typename Component>
		FORCEINLINE bool HasComponent(EntityHandle entity)
		{
			return registry.all_of<Component>(entity);
		}

		template<typename... Component>
		FORCEINLINE bool HasAllComponents(EntityHandle entity)
		{
			return registry.all_of<Component ...>(entity);
		}

		template<typename... Component>
		FORCEINLINE bool HasAnyComponent(EntityHandle entity)
		{
			return registry.any_of<Component ...>(entity);
		}

		template<typename Component, typename... Args>
		FORCEINLINE Component& AddComponent(EntityHandle entity, Args&&... args)
		{
			return registry.emplace<Component>(entity, std::forward<Args>(args)...);
		}

		template<typename Component, typename... Args>
		FORCEINLINE void AddOrReplaceComponent(EntityHandle entity, Args&&... args)
		{
			registry.emplace_or_replace<Component>(entity, std::forward<Args>(args)...);
		}

		template<typename Component, typename... Args>
		FORCEINLINE void ReplaceComponent(EntityHandle entity, Args&&... args)
		{
			registry.replace<Component>(entity, std::forward<Args>(args)...);
		}

		template<typename Component>
		FORCEINLINE Component& GetComponent(EntityHandle entity) const
		{
			return registry.get<Component>(entity);
		}

		template<typename Component>
		FORCEINLINE Component& GetComponent(EntityHandle entity)
		{
			return registry.get<Component>(entity);
		}

		template<typename... Component>
		FORCEINLINE decltype(auto) GetComponents(EntityHandle entity) const
		{
			return registry.get<Component ...>(entity);
		}

		template<typename... Component>
		FORCEINLINE decltype(auto) GetComponents(EntityHandle entity)
		{
			return registry.get<Component ...>(entity);
		}

		template<typename Component, typename... Args>
		FORCEINLINE Component& GetOrAddComponent(EntityHandle entity, Args&&... args)
		{
			return registry.get_or_emplace<Component>(entity, std::forward<Args>(args)...);
		}

		template<typename Component>
		FORCEINLINE Component* TryGetComponent(EntityHandle entity)
		{
			return registry.try_get<Component>(entity);
		}

		template<typename... Component>
		FORCEINLINE auto TryGetComponents(EntityHandle entity)
		{
			return registry.try_get<Component ...>(entity);
		}

		template<typename Component>
		FORCEINLINE auto RemoveComponent(EntityHandle entity)
		{
			return registry.remove<Component>(entity);
		}

	private:
		entt::registry registry;
	};
}
