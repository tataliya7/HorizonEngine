#include "Entity.h"
#include "NameComponent.h"
#include "TransformComponent.h"
#include "SceneHierarchyComponent.h"

namespace HE
{
	EntityHandle EntityManager::CreateEntity(const char* name)
	{
		EntityHandle entity(registry.create());
		AddComponent<NameComponent>(entity, name);
		AddComponent<TransformComponent>(entity);
		AddComponent<SceneHierarchyComponent>(entity);
		return entity;
	}

	void EntityManager::DestroyEntity(EntityHandle entity)
	{
		registry.destroy(entity);
	}

	void EntityManager::Clear()
	{
		registry.clear();
	}

	bool EntityManager::HasEntity(EntityHandle entity)
	{
		return registry.valid(entity);
	}
}