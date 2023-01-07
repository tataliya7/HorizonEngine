#include "HorizonExample.h"
#include "HorizonExampleFirstPersonCameraController.h"
#include "Daisy/DaisyRenderer.h"

import HorizonEngine.Render.VulkanRenderBackend;

namespace HE
{
	HorizonExample::HorizonExample()
	{
		name = HORIZON_EXAMPLE_NAME;
	}

	HorizonExample::~HorizonExample()
	{
		
	}

	void HorizonExample::Setup()
	{
		// Create default scene
		scene = SceneManager::CreateScene("DefaultScene");
		SceneManager::SetActiveScene(scene);

		// Create main camera
		mainCamera = scene->GetEntityManager()->CreateEntity("MainCamera");
		{
			// When an entity is added to the scene, TransformComponent and SceneHierarchyComponent are automatically created
			auto& transformComponent = scene->GetEntityManager()->GetComponent<TransformComponent>(mainCamera);
			transformComponent.position = Vector3(0.0f, 0.0f, 5.0f);
			transformComponent.rotation = Vector3(0.0f, 0.0f, 0.0f);

			auto& cameraComponent = scene->GetEntityManager()->AddComponent<CameraComponent>(mainCamera);
			cameraComponent.type = CameraType::Perpective;
			cameraComponent.nearPlane = 0.5f;
			cameraComponent.farPlane = 1000.0f;
			cameraComponent.fieldOfView = 60.0f;
			cameraComponent.aspectRatio = 16.0f / 9.0f;
			cameraComponent.overrideAspectRatio = false;

			// Attach camera controller to main camera
			scene->GetEntityManager()->AddComponent<ScriptComponent>(mainCamera).Bind<HorizonExampleFirstPersonCameraController>();
		}

		// Create directional light
		EntityHandle directionalLight = scene->GetEntityManager()->CreateEntity("DirectionalLight");
		{
			auto& directionalLightComponent = scene->GetEntityManager()->AddComponent<DirectionalLightComponent>(directionalLight);
			directionalLightComponent.color = Vector4(1.0f);
			directionalLightComponent.intensity = 1.0f;
		}

		// Create sky light
		EntityHandle skyLight = scene->GetEntityManager()->CreateEntity("SkyLight");
		{
			auto& skyLightComponent = scene->GetEntityManager()->AddComponent<SkyLightComponent>(skyLight);
			skyLightComponent.cubemapResolution = 1024;
			skyLightComponent.SetCubemap("../../../Assets/HDRIs/HDR_029_Sky_Cloudy_Ref.hdr");
		}

		// Create meshes
		for (uint32 i = 0; i < 16; i++)
		{
			EntityHandle box = scene->GetEntityManager()->CreateEntity("Box");
			auto& transformComponent = scene->GetEntityManager()->GetComponent<TransformComponent>(box);
			transformComponent.position = Vector3(2.0f * i - 16.0, 20.0f, 11.0f);
			auto& staticMeshComponent = scene->GetEntityManager()->AddComponent<StaticMeshComponent>(box);
			staticMeshComponent.meshSource = "../../../Assets/Models/Box/Box.gltf";
			auto& rigidBodyComponent = scene->GetEntityManager()->AddComponent<RigidBodyComponent>(box);
			rigidBodyComponent.type = RigidBodyComponent::Type::Dynamic;
			auto& doxColliderComponent = scene->GetEntityManager()->AddComponent<BoxColliderComponent>(box);
		}

		"D:/Programming/HorizonEngine/Assets/Audio/HORIZON.mp3"
		
		// TODO: Remove this
		scene->renderScene = new RenderScene();
		scene->physicsScene = new PhysicsScene();
		scene->renderScene->Setup(scene, daisyRenderer);

		// Create scene view
		view = new SceneView();
	}

	void HorizonExample::Clear()
	{
		SceneManager::DestroyScene(scene);

		delete view;
	}

	void HorizonExample::OnUpdate(float deltaTime)
	{
		OPTICK_EVENT();

		scene->physicsScene->Simulate(deltaTime);
		scene->Update(deltaTime);
		scene->renderScene->Update(deltaTime);
	}

	void HorizonExample::OnRender()
	{
		OPTICK_EVENT();

		auto& cameraComponent = scene->GetEntityManager()->GetComponent<CameraComponent>(mainCamera);
		auto& transformComponent = scene->GetEntityManager()->GetComponent<TransformComponent>(mainCamera);
		view->scene = scene->renderScene;
		view->target = RenderBackendGetActiveSwapChainBuffer(renderBackend, swapChain);
		view->targetDesc = RenderBackendTextureDesc::Create2D(swapChainWidth, swapChainHeight, PixelFormat::BGRA8Unorm, TextureCreateFlags::Present);
		view->targetWidth = swapChainWidth;
		view->targetHeight = swapChainHeight;
		view->camera.fieldOfView = cameraComponent.fieldOfView;
		view->camera.zNear = cameraComponent.nearPlane;
		view->camera.zFar = cameraComponent.farPlane;
		if (cameraComponent.overrideAspectRatio)
		{
			view->camera.aspectRatio = cameraComponent.aspectRatio;
		}
		else
		{
			view->camera.aspectRatio = (float)swapChainWidth / (float)swapChainHeight;
		}
		view->camera.position = transformComponent.position;
		view->camera.euler = transformComponent.rotation;

		daisyRenderer->Render(view);
	}

	void HorizonExample::OnDrawUI()
	{

	}
}