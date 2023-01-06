module;

#include <ECS/ECS.h>

module HorizonEngine.SceneManagement;

namespace HE
{
	uint32 SceneManager::LoadedSceneCount = 0;
	Scene* SceneManager::ActiveScene = nullptr;
	std::map<std::string, std::shared_ptr<Scene>> SceneManager::SceneMapByName;

	Scene* SceneManager::CreateScene(const std::string& name)
	{
		auto scene = new Scene();
		scene->name = name;
		SceneMapByName[name] = (std::shared_ptr<Scene>)scene;
		return scene;
	}

	void SceneManager::DestroyScene(Scene* scene)
	{
		std::string name = scene->name;
		SceneMapByName[name] = nullptr;
	}

	Scene* SceneManager::GetActiveScene()
	{
		return ActiveScene;
	}

	Scene* SceneManager::GetSceneByName(const std::string& name)
	{
		if (SceneMapByName.find(name) == SceneMapByName.end())
		{
			return nullptr;
		}
		return SceneMapByName[name].get();
	}

	void SceneManager::SetActiveScene(Scene* scene)
	{
		ActiveScene = scene;
	}

	void SceneManager::LoadScene(const std::string& name)
	{
		if (SceneMapByName.find(name) == SceneMapByName.end())
		{
			return;
		}
	}

	void SceneManager::LoadSceneAsync(const std::string& name)
	{
		// TODO
	}

	void SceneManager::UnloadSceneAsync(Scene* scene)
	{
		// TODO
	}

	void SceneManager::MergeScenes(Scene* dstScene, Scene* srcScene)
	{
		// TODO
	}

	Scene::Scene()
	{
		entityManager = new EntityManager();
		entityManager->OnConstruct<TransformComponent>().connect<&entt::registry::emplace_or_replace<TransformDirtyComponent>>();
		entityManager->OnUpdate<TransformComponent>().connect<&entt::registry::emplace_or_replace<TransformDirtyComponent>>();
	}

	Scene::~Scene()
	{
		delete entityManager;
	}

	static void UpdateTransform(EntityManager* manager, EntityHandle entity)
	{
		const auto& hierarchy = manager->GetComponent<SceneHierarchyComponent>(entity);
		auto& transform = manager->GetComponent<TransformComponent>(entity);
		transform.Update();
		auto currentEntity = hierarchy.first;
		for (uint32 i = 0; i < hierarchy.numChildren; i++)
		{
			if (manager->HasComponent<TransformDirtyComponent>(currentEntity))
			{
				continue;
			}
			UpdateTransform(manager, currentEntity);
			currentEntity = manager->GetComponent<SceneHierarchyComponent>(currentEntity).next;
		}
		manager->RemoveComponent<TransformDirtyComponent>(entity);
	}

	void Scene::Update(float deltaTime)
	{
		// Update scripts
		{
			entityManager->GetView<ScriptComponent>().each([&](EntityHandle entity, auto& component)
			{
				if (component.scriptable == nullptr)
				{
					component.ConstructorFunc();
					component.scriptable->manager = entityManager;
					component.scriptable->entity = entity;
					if (component.OnCreateFunc)
					{
						component.OnCreateFunc(component.scriptable);
					}
				}
				if (component.OnUpdateFunc)
				{
					component.OnUpdateFunc(component.scriptable, deltaTime);
				}
			});
		}

		// Update animations

		// Update transforms
		entityManager->Get()->sort<TransformDirtyComponent>([&](EntityHandle lhs, EntityHandle rhs)
		{
			const auto& lc = entityManager->GetComponent<SceneHierarchyComponent>(lhs);
			const auto& rc = entityManager->GetComponent<SceneHierarchyComponent>(rhs);
			return lc.depth < rc.depth;
		});

		entityManager->GetView<TransformDirtyComponent>().each([&](EntityHandle entity)
		{
			UpdateTransform(entityManager, entity);
		});
	}
}