#include "HorizonExample.h"
#include "HorizonExampleFirstPersonCameraController.h"

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
		// Create scene
		scene = SceneManager::CreateScene("ExampleScene");
		SceneManager::SetActiveScene(scene);

		scene->SetShouldSimulate(true);
		scene->SetShouldUpdateScripts(true);

		// Create main camera
		mainCamera = scene->CreateEntity("MainCamera");
		{
			// When an entity is added to the scene, TransformComponent and SceneHierarchyComponent are automatically created
			auto& transformComponent = scene->GetEntityManager()->GetComponent<TransformComponent>(mainCamera);
			transformComponent.position = Vector3(0.0f, 0.0f, 5.0f);
			transformComponent.rotation = Vector3(0.0f, 0.0f, 0.0f);

			CameraComponent cameraComponent;
			cameraComponent.type = CameraComponent::Type::Perpective;
			cameraComponent.nearPlane = 0.5f;
			cameraComponent.farPlane = 1000.0f;
			cameraComponent.fieldOfView = 60.0f;
			cameraComponent.aspectRatio = 16.0f / 9.0f;
			cameraComponent.overrideAspectRatio = false;
			scene->GetEntityManager()->AddComponent<CameraComponent>(mainCamera, cameraComponent);

			// Attach camera controller to main camera
			scene->GetEntityManager()->AddComponent<ScriptComponent>(mainCamera).Bind<HorizonExampleFirstPersonCameraController>();
		}

		// Create main light
		EntityHandle directionalLight = scene->CreateEntity("DirectionalLight");
		{
			DirectionalLightComponent directionalLightComponent;
			directionalLightComponent.color = Vector4(1.0f);
			directionalLightComponent.intensity = 1.0f;
			scene->GetEntityManager()->AddComponent<DirectionalLightComponent>(directionalLight, directionalLightComponent);
		}

		// Create sky light
		EntityHandle skyLight = scene->CreateEntity("SkyLight");
		{
			SkyLightComponent skyLightComponent;
			skyLightComponent.cubemapResolution = 1024;
			skyLightComponent.SetCubemap("../../../Assets/HDRIs/HDR_029_Sky_Cloudy_Ref.hdr");
			scene->GetEntityManager()->AddComponent<SkyLightComponent>(skyLight, skyLightComponent);
		}

		// Create meshes
		for (uint32 i = 0; i < 16; i++)
		{
			EntityHandle box = scene->CreateEntity("Box");

			auto& transformComponent = scene->GetEntityManager()->GetComponent<TransformComponent>(box);
			transformComponent.position = Vector3(2.0f * i - 16.0, 20.0f, 11.0f);

			StaticMeshComponent staticMeshComponent;
			staticMeshComponent.meshSource = "../../../Assets/Models/Box/Box.gltf";
			scene->GetEntityManager()->AddComponent<StaticMeshComponent>(box, staticMeshComponent);

			RigidBodyComponent rigidBodyComponent;
			rigidBodyComponent.type = RigidBodyComponent::Type::Dynamic;
			rigidBodyComponent.SetBoxCollider(Vector3(1.0f), Vector3(0.0f));
			scene->GetEntityManager()->AddComponent<RigidBodyComponent>(box, rigidBodyComponent);
		}

		// TODO: Remove this
		scene->GetRenderScene()->Setup();

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

		scene->Update(deltaTime);
	}

	void HorizonExample::OnRender()
	{
		OPTICK_EVENT();

		auto& cameraComponent = scene->GetEntityManager()->GetComponent<CameraComponent>(mainCamera);
		auto& transformComponent = scene->GetEntityManager()->GetComponent<TransformComponent>(mainCamera);
		view->scene = scene->GetRenderScene();
		view->target = RenderBackendGetActiveSwapChainBuffer(renderBackend, swapChain);
		view->targetDesc = RenderBackendTextureDesc::Create2D(swapChainWidth, swapChainHeight, PixelFormat::BGRA8Unorm, TextureCreateFlags::Present);
		view->targetWidth = swapChainWidth;
		view->targetHeight = swapChainHeight;
		view->camera.fieldOfView = cameraComponent.fieldOfView;
		view->camera.zNear = cameraComponent.nearPlane;
		view->camera.zFar = cameraComponent.farPlane;
		view->frameIndex = (uint32)GetFrameCounter();
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