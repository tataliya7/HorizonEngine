module;

#include "Core/CoreDefinitions.h"
#include "AssimpImporter/AssimpImporter.h"

module HorizonEngine.Scene;

namespace HE
{
	Scene::Scene(const std::string& name)
		: name(name)
		, hasSkyLight(false)
	{
		entityManager = new EntityManager();

		entityManager->OnConstruct<TransformComponent>().connect<&entt::registry::emplace_or_replace<TransformDirtyComponent>>();
		entityManager->OnUpdate<TransformComponent>().connect<&entt::registry::emplace_or_replace<TransformDirtyComponent>>();

		entityManager->OnConstruct<DirectionalLightComponent>().connect<&Scene::OnDirectionalLightComponentConstruct>(this);
		entityManager->OnDestroy<DirectionalLightComponent>().connect<&Scene::OnDirectionalLightComponentDestroy>(this);

		entityManager->OnConstruct<SkyLightComponent>().connect<&Scene::OnSkyLightComponentConstruct>(this);
		entityManager->OnDestroy<SkyLightComponent>().connect<&Scene::OnSkyLightComponentDestroy>(this);

		entityManager->OnConstruct<StaticMeshComponent>().connect<&Scene::OnStaticMeshComponentConstruct>(this);
		entityManager->OnDestroy<StaticMeshComponent>().connect<&Scene::OnStaticMeshComponentDestroy>(this);

		entityManager->OnConstruct<RigidBodyComponent>().connect<&Scene::OnRigidBodyComponentConstruct>(this);
		entityManager->OnDestroy<RigidBodyComponent>().connect<&Scene::OnRigidBodyComponentDestroy>(this);

		physicsScene = new PhysicsScene();
		renderScene = new RenderScene();
	}

	Scene::~Scene()
	{
		delete physicsScene;
		delete renderScene;
		delete entityManager;
	}

	void Scene::Clear()
	{
		entityManager->Clear();
	}

	void Scene::OnDirectionalLightComponentConstruct(entt::registry& registry, entt::entity entity)
	{
		auto& component = entityManager->GetComponent<DirectionalLightComponent>(entity);
		component.proxy = new LightRenderProxy(&component);
		renderScene->SetMainLight(component.proxy);
	}

	void Scene::OnDirectionalLightComponentDestroy(entt::registry& registry, entt::entity entity)
	{

	}

	void Scene::OnSkyLightComponentConstruct(entt::registry& registry, entt::entity entity)
	{
		if (!hasSkyLight)
		{
			auto& component = entityManager->GetComponent<SkyLightComponent>(entity);
			component.proxy = new SkyLightRenderProxy(&component);
			renderScene->SetSkyLight(component.proxy);
			hasSkyLight = true;
		}
		else
		{
			ASSERT(!hasSkyLight && "Can't create more than one SkyLightComponent!");
		}
	}
	
	void Scene::OnSkyLightComponentDestroy(entt::registry& registry, entt::entity entity)
	{
		
	}

	void Scene::OnStaticMeshComponentConstruct(entt::registry& registry, entt::entity entity)
	{
		auto& transformComponent = entityManager->GetComponent<TransformComponent>(entity);
		auto& staticMeshComponent = entityManager->GetComponent<StaticMeshComponent>(entity);

		AssimpImporter assimpImporter;
		assimpImporter.ImportAsset(staticMeshComponent.meshSource.c_str());

		Mesh* meshSource = AssetManager::GetAsset<Mesh>(staticMeshComponent.meshSource);

		MeshRenderProxy* proxy = new MeshRenderProxy(&staticMeshComponent);
		proxy->worldMatrix = transformComponent.world;
		proxy->mesh = meshSource;
		staticMeshComponent.proxy = proxy;

		renderScene->meshes.push_back(proxy);
	}

	void Scene::OnStaticMeshComponentDestroy(entt::registry& registry, entt::entity entity)
	{

	}

	void Scene::OnRigidBodyComponentConstruct(entt::registry& registry, entt::entity entity)
	{
		physicsScene->CreateActor(entityManager, entity);
	}

	void Scene::OnRigidBodyComponentDestroy(entt::registry& registry, entt::entity entity)
	{

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
        if (ShouldUpdateScripts())
        {
            // Update scripts
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
        
		if (ShouldSimulate())
		{
			physicsScene->Simulate(deltaTime);
		}

		// Update animation
		{

		}

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

		// Update main camera
		{

		}

		// Update directional lights
		{
			entityManager->GetView<DirectionalLightComponent>().each([&](EntityHandle entity, auto& directionalLight)
			{
				const auto& transform = entityManager->GetComponent<TransformComponent>(entity);
				directionalLight.proxy->direction = Math::Normalize(Vector3(Quaternion(Math::DegreesToRadians(transform.rotation)) * Vector4(0.0, 0.0, -1.0, 0.0)));
			});
		}

		// Update meshes
		{
			entityManager->GetView<StaticMeshComponent>().each([&](EntityHandle entity, auto& component)
			{
				auto& transform = entityManager->GetComponent<TransformComponent>(entity);
				component.proxy->worldMatrix = transform.world;
			});
		}

		renderScene->Update(deltaTime);

		// Update audio sources and listeners
		{

		}
	}
}