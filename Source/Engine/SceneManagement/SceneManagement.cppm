module;

#include <string>
#include <map>
#include <string>
#include <memory>
#include <filesystem>

#include "ECS/ECS.h"

export module HorizonEngine.SceneManagement;

import HorizonEngine.Core;

export namespace HE
{
	class RenderScene;
	class PhysicsScene;

	/**
	 * Run-time data structure for *.horizon file.
	 */
	class Scene
	{
	public:
		Scene();
		~Scene();
		void Update(float deltaTime);
		std::string name;
		std::string url;
		bool isLoaded;
		bool isDirty;

		int frame = 0;
		EntityManager* GetEntityManager()
		{
			return entityManager;
		}
		RenderScene* renderScene;
		PhysicsScene* physicsScene;
	private:
		friend class SceneSerializer;
		Guid guid;
		EntityManager* entityManager;
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