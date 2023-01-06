#include "HorizonExample.h"
#include "HorizonExampleFirstPersonCameraController.h"
#include "CameraAnimation.h"
#include "Daisy/DaisyRenderer.h"

import HorizonEngine.Render.VulkanRenderBackend;
import HorizonEngine.Input;

namespace HE
{
	class CameraController : public HorizonExampleFirstPersonCameraController
	{
	public:

		CameraController() = default;

		virtual ~CameraController() = default;

		CameraAnimationSequence cameraAnim;

		void OnCreate()
		{
			cameraAnim.Reset();
			cameraAnim.timeLengthInSeconds = 30.0f;

			cameraAnim.AddPositionSample(0.0f, HE::Vector3(-10.123f, 4.507f, 5.226f));
			cameraAnim.AddRotationSample(0.0f, HE::Vector3(-11.149f, 0.000f, -155.316f));

			cameraAnim.AddPositionSample(5.0f, HE::Vector3(-9.651f, -0.032f, 4.861f));
			cameraAnim.AddRotationSample(5.0f, HE::Vector3(-4.552f, 0.000f, -161.695f));

			cameraAnim.AddPositionSample(10.0f, HE::Vector3(-7.590f, -2.993f, 4.607f));
			cameraAnim.AddRotationSample(10.0f, HE::Vector3(-0.625f, 0.000f, -107.049f));

			cameraAnim.AddPositionSample(15.0f, HE::Vector3(-4.391f, -1.828f, 4.995f));
			cameraAnim.AddRotationSample(15.0f, HE::Vector3(-5.293f, 0.000f, -46.153f));

			cameraAnim.AddPositionSample(20.0f, HE::Vector3(-1.339f, 0.214f, 4.020f));
			cameraAnim.AddRotationSample(20.0f, HE::Vector3(-31.321f, 0.000f, -68.891f));

			cameraAnim.AddPositionSample(25.0f, HE::Vector3(1.773f, 0.486f, 2.424f));
			cameraAnim.AddRotationSample(25.0f, HE::Vector3(-21.487f, 0.000f, -90.298f));

			cameraAnim.AddPositionSample(30.0f, HE::Vector3(6.927f, 0.278f, 1.341f));
			cameraAnim.AddRotationSample(30.0f, HE::Vector3(0.703f, 0.000f, -90.552f));

			cameraAnim.InitCurve();
		}

		void OnDestroy()
		{

		}

		void OnUpdate(float deltaTime)
		{
			auto& tranform = GetComponent<HE::TransformComponent>();

			////if (Input::GetKeyDown(KeyCode::R))
			////{
			////	printf("cameraAnim.AddPositionSample(%.3f, HE::Vector3(%.3ff, %.3ff, %.3ff));\n", 0.0f, tranform.position.x, tranform.position.y, tranform.position.z);
			////	printf("cameraAnim.AddPositionSample(%.3f, HE::Vector3(%.3ff, %.3ff, %.3ff));\n", 0.0f, tranform.rotation.x, tranform.rotation.y, tranform.rotation.z);
			////	printf("\n");
			////	//HE_LOG_INFO("cameraAnim.AddPositionSample({:.3f}, HE::Vector3({:.3f}, {:.3f}, {:.3f}))", 0.0f, tranform.position.x, tranform.position.y, tranform.position.z);
			//////	HE_LOG_INFO("cameraAnim.AddRotationSample({:.3f}, HE::Vector3({:.3f}, {:.3f}, {:.3f}))", 0.0f, tranform.rotation.x, tranform.rotation.y, tranform.rotation.z);
			////}

			if (false)
			{
				float targetFPS = 30.0f;
				cameraAnim.UpdateCameraMotion(1.0f / targetFPS, tranform.position, tranform.rotation);
			}
			else
			{
				UpdateTransform(deltaTime, tranform.position, tranform.rotation);
			}
		}
	};

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
			scene->GetEntityManager()->AddComponent<ScriptComponent>(mainCamera).Bind<CameraController>();
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

		// Create model
		EntityHandle model = scene->GetEntityManager()->CreateEntity("Model");
		{
			auto& staticMeshComponent = scene->GetEntityManager()->AddComponent<StaticMeshComponent>(model);
			staticMeshComponent.meshSource = "../../../Assets/Models/Sponza/glTF/Sponza.gltf";
			//staticMeshComponent.meshSource = "../../../Assets/Models/EnvironmentTest/glTF-IBL/EnvironmentTest.gltf";
			//staticMeshComponent.meshSource = "../../../Assets/Models/SciFiHelmet/glTF/SciFiHelmet.gltf";
			//staticMeshComponent.meshSource = "../../../Assets/Models/DamagedHelmet/glTF/DamagedHelmet.gltf";
			//staticMeshComponent.meshSource = "../../../Assets/Models/SunTemple_v4/SunTemple.gltf";
		}

		selectedEntity = model;

		// TODO: Remove this
		scene->renderScene = new RenderScene();
		scene->physicsScene = new PhysicsScene();
		scene->renderScene->Setup(scene, daisyRenderer);

		// Create scene view
		view = new SceneView();
		view->captureTargetDescs[0] = RenderBackendTextureDesc::Create2D(1280, 720, PixelFormat::RGBA32Float, TextureCreateFlags::Readback);
		view->captureTargets[0] = RenderBackendCreateTexture(renderBackend, ~0u, &view->captureTargetDescs[0], nullptr, "CaptureTarget0");
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

		bool captureFrame = true;
		if (captureFrame)
		{
			// RenderBackendFlushRenderDevices(renderBackend);
			/*	String outputPath = ASSETS_PATH + "screenshots/" + std::to_string(frameID) + ".png";
			readbackBuffer->Map();
			if (!stbi_write_png(outputPath.c_str(), displayResolution.width, displayResolution.height, 4, readbackBuffer->GetData(), 0))
			{
				LOG_ERROR("Failed to save screenshot.");
			}
			readbackBuffer->Unmap();*/
		}
	}
}