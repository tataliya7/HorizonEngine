module;

#include <string>
#include <memory>

module HorizonEngine.Scene;

namespace HE
{
	uint32 SceneManager::LoadedSceneCount = 0;
	Scene* SceneManager::ActiveScene = nullptr;
	std::map<std::string, std::shared_ptr<Scene>> SceneManager::SceneMapByName;

	Scene* SceneManager::CreateScene(const std::string& name)
	{
		auto scene = new Scene(name);
		SceneMapByName[name] = (std::shared_ptr<Scene>)scene;
		return scene;
	}

	void SceneManager::DestroyScene(Scene* scene)
	{
		std::string name = scene->GetName();
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
}