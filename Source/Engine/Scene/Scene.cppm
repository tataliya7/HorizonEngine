module;

#include "Core/CoreDefinitions.h"
#include <entt/entt.hpp>

export module HorizonEngine.Scene;

import <string>;
import <filesystem>;
import HorizonEngine.Core;
import HorizonEngine.Entity;
import HorizonEngine.Physics;
import HorizonEngine.Render;

export namespace HE
{
	class Scene
	{
	public:
		
		Scene(const std::string& name);
		~Scene();

		const std::string& GetName() const
		{
			return name;
		}

		EntityHandle CreateEntity(const char* name)
		{
			EntityHandle entity = entityManager->CreateEntity();
			entityManager->AddComponent<NameComponent>(entity, name);
			entityManager->AddComponent<TransformComponent>(entity);
			entityManager->AddComponent<SceneHierarchyComponent>(entity);
			return entity;
		}

		void DestroyEntity(EntityHandle entity)
		{
			entityManager->DestroyEntity(entity);
		}

		EntityManager* GetEntityManager()
		{
			return entityManager;
		}

		PhysicsScene* GetPhysicsScene()
		{
			return physicsScene;
		}

		RenderScene* GetRenderScene()
		{
			return renderScene;
		}
		
		bool ShouldSimulate() const
		{
			return shouldSimulate;
		}

		bool ShouldUpdateScripts() const
		{
			return shouldUpdateScripts;
		}

		void SetShouldSimulate(bool value)
		{
			shouldSimulate = value;
		}

		void SetShouldUpdateScripts(bool value)
		{
			shouldUpdateScripts = value;
		}

		void Update(float deltaTime);

		void Clear();

		void OnDirectionalLightComponentConstruct(entt::registry& registry, entt::entity entity);
		void OnDirectionalLightComponentDestroy(entt::registry& registry, entt::entity entity);

		void OnSkyLightComponentConstruct(entt::registry& registry, entt::entity entity);
		void OnSkyLightComponentDestroy(entt::registry& registry, entt::entity entity);

		void OnStaticMeshComponentConstruct(entt::registry& registry, entt::entity entity);
		void OnStaticMeshComponentDestroy(entt::registry& registry, entt::entity entity);

		void OnRigidBodyComponentConstruct(entt::registry& registry, entt::entity entity);
		void OnRigidBodyComponentDestroy(entt::registry& registry, entt::entity entity);

	private:
		friend class SceneSerializer;
		std::string name;
		EntityManager* entityManager;
		PhysicsScene* physicsScene;
		RenderScene* renderScene;
		bool shouldSimulate;
		bool shouldUpdateScripts;
		bool hasSkyLight;
	};


	/**
	 * Scene management at runtime.
	 */
	class SceneManager
	{
	public:
		static uint32 LoadedSceneCount;
		static Scene* ActiveScene;
		static std::map<std::string, std::shared_ptr<Scene>> SceneMapByName;
		static Scene* CreateScene(const std::string& name);
		static void DestroyScene(Scene* scene);
		static Scene* GetActiveScene();
		static Scene* GetSceneByName(const std::string& name);
		static void SetActiveScene(Scene* scene);
		static void LoadScene(const std::string& name);
		static void LoadSceneAsync(const std::string& name);
		static void UnloadSceneAsync(Scene* scene);
		static void MergeScenes(Scene* dstScene, Scene* srcScene);
	};

	class SceneSerializer
	{
	public:
		inline static std::string_view DefaultExtension = ".horizon";
		SceneSerializer(Scene* scene) : scene(scene) {}
		void Serialize(const std::filesystem::path& filename);
		bool Deserialize(const std::filesystem::path& filename);
	private:
		Scene* scene;
	};
}