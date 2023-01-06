#include "ModelViewer.h"
#include "MainWindow.h"
#include "HybridRenderPipeline/HybridRenderPipeline.h"

ModelViewerApp* ModelViewerApp::Instance = nullptr;

ModelViewerApp::ModelViewerApp()
	: isExitRequested(false)
	, frameCounter(0)
{
	ASSERT(!Instance);
	Instance = this;
}

ModelViewerApp::~ModelViewerApp()
{
	ASSERT(Instance);
	Instance = nullptr;
}

void SetupEarthAtmosphere(HE::AtmosphereParameters* outAtmosphere)
{
	// Values shown here are the result of integration over wavelength power spectrum integrated with paricular function.
	// Refer to https://github.com/ebruneton/precomputed_atmospheric_scattering for details.

	// All units in kilometers
	const float EarthBottomRadius = 6360.0f;
	const float EarthTopRadius = 6460.0f;   // 100km atmosphere radius, less edge visible and it contain 99.99% of the atmosphere medium https://en.wikipedia.org/wiki/K%C3%A1rm%C3%A1n_line
	const float EarthRayleighScaleHeight = 8.0f;
	const float EarthMieScaleHeight = 1.2f;

	const double maxSunZenithAngle = M_PI * 120.0 / 180.0;
	*outAtmosphere = HE::AtmosphereParameters{
		// Earth
		.bottomRadius = EarthBottomRadius,
		.topRadius = EarthTopRadius,
		.groundAlbedo = { 0.0f, 0.0f, 0.0f },
		// Raleigh scattering
		.rayleighDensity = { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, -1.0f / EarthRayleighScaleHeight, 0.0f, 0.0f } },
		.rayleighScattering = { 0.005802f, 0.013558f, 0.033100f }, // 1/km
		// Mie scattering
		.mieDensity = { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, -1.0f / EarthMieScaleHeight, 0.0f, 0.0f } },
		.mieScattering = { 0.003996f, 0.003996f, 0.003996f }, // 1/km
		.mieExtinction = { 0.004440f, 0.004440f, 0.004440f }, // 1/km
		.miePhaseFunctionG = 0.8f,
		// Ozone absorption
		.absorptionDensity = { { 25.0f, 0.0f, 0.0f, 1.0f / 15.0f, -2.0f / 3.0f }, { 0.0f, 0.0f, 0.0f, -1.0f / 15.0f, 8.0f / 3.0f } },
		.absorptionExtinction = { 0.000650f, 0.001881f, 0.000085f }, // 1/km
		.cosMaxSunZenithAngle = (float)HE::Math::Cos(maxSunZenithAngle),
	};
}

bool ModelViewerApp::Init()
{
	using namespace HE;

	arena = new LinearArena(nullptr, 1048576);

	uint32 initialWindowWidth = 1920;
	uint32 initialWindowHeight = 1080;
	
	GLFWInit();
	WindowCreateInfo windowInfo = {
		.width = initialWindowWidth,
		.height = initialWindowHeight,
		.title = HE_APPLICATION_NAME,
		.flags = WindowFlags::Resizable
	};
	window = new MainWindow(&windowInfo);
	Input::SetCurrentContext(window->handle);

	shaderCompiler = CreateDxcShaderCompiler();

	int flags = VULKAN_RENDER_BACKEND_CREATE_FLAGS_VALIDATION_LAYERS | VULKAN_RENDER_BACKEND_CREATE_FLAGS_SURFACE;
	renderBackend = VulkanRenderBackendCreateBackend(flags);
	
	uint32 deviceMask;
	uint32 physicalDeviceID = 0;
	RenderBackendCreateRenderDevices(renderBackend, &physicalDeviceID, 1, &deviceMask);
	
	swapChain = RenderBackendCreateSwapChain(renderBackend, deviceMask, (uint64)window->GetNativeHandle());
	swapChainWidth = window->GetWidth();
	swapChainHeight = window->GetHeight();

	uiRenderer = new UIRenderer(window->handle, renderBackend, shaderCompiler);
	uiRenderer->Init();

	scene = new Scene();

	GLTF2ImportSettings settings;
	ImportGLTF2("../../Assets/Models/DamagedHelmet/glTF/DamagedHelmet.gltf", settings, scene);
	//ImportGLTF2("../../Assets/Models/Sponza/glTF/Sponza.gltf", settings, scene);
	scene->renderBackend = renderBackend;
	scene->UploadResources();

	auto entityManager = scene->GetEntityManager();

	auto directionalLight = entityManager->CreateEntity("Directional Light");
	LightComponent lightComponent = {
		.type = LightType::Directional,
		.color = Vector4(1.0f),
		.intensity = 1.0f,
		.direction = Vector4(),
	};
	entityManager->AddComponent<LightComponent>(directionalLight, lightComponent);
	entityManager->AddComponent<TransformComponent>(directionalLight);
	entityManager->AddComponent<HierarchyComponent>(directionalLight);

	CameraComponent cameraComponent = {
		.type = CameraType::Perpective,
		.nearPlane = 0.1,
		.farPlane = 3000.0,
		.fieldOfView = 60.0,
	};
	
	TransformComponent cameraTransform = {
		.position = Vector3(5.0, 0.0, 0.0),
		.rotation = Vector3(0.0, 0.0, 0.0),
		.scale = Vector3(1.0),
	};
	mainCamera = entityManager->CreateEntity("Main Camera");
	entityManager->AddComponent<CameraComponent>(mainCamera, cameraComponent);
	entityManager->AddComponent<TransformComponent>(mainCamera, cameraTransform);
	entityManager->AddComponent<HierarchyComponent>(mainCamera);

	auto sky = entityManager->CreateEntity("Sky");
	entityManager->AddComponent<TransformComponent>(sky);
	entityManager->AddComponent<HierarchyComponent>(sky);
	auto& skyAtmosphereComponent = entityManager->AddComponent<SkyAtmosphereComponent>(sky);
	SetupEarthAtmosphere(&skyAtmosphereComponent.atmosphere);
	skyAtmosphereComponent.viewRayMarchMinSPP = 4;
	skyAtmosphereComponent.viewRayMarchMaxSPP = 16;
	skyAtmosphereComponent.multipleScatteringFactor = 1.0;

	renderContext = new RenderContext();
	renderContext->arena = arena;
	renderContext->renderBackend = renderBackend;
	renderContext->shaderCompiler = shaderCompiler;
	renderContext->uiRenderer = uiRenderer;

	renderPipeline = new HybridRenderPipeline(renderContext);
	renderPipeline->Init();

	sceneView = new SceneView();
	sceneView->scene = scene;

	return true;
}

void ModelViewerApp::Exit()
{
	delete scene;
	uiRenderer->Shutdown();
	delete uiRenderer;
	VulkanRenderBackendDestroyBackend(renderBackend);
	DestroyDxcShaderCompiler(shaderCompiler);
	delete window;
	GLFWExit();
	delete arena;
}

void ModelViewerApp::Update(float deltaTime)
{
	auto& cameraTransform = scene->GetEntityManager()->GetComponent<HE::TransformComponent>(mainCamera);
	cameraController.Update(deltaTime, cameraTransform.position, cameraTransform.rotation);
	
	scene->Update();
}

void ModelViewerApp::Render()
{
	uiRenderer->BeginFrame();
	OnImGui();
	uiRenderer->EndFrame();

	auto& camera = scene->GetEntityManager()->GetComponent<HE::CameraComponent>(mainCamera);
	camera.aspectRatio = (float)swapChainWidth / (float)swapChainHeight;
	auto& cameraTransform = scene->GetEntityManager()->GetComponent<HE::TransformComponent>(mainCamera);
	sceneView->renderPipeline = renderPipeline;
	sceneView->target = HE::RenderBackendGetActiveSwapChainBuffer(renderBackend, swapChain);
	sceneView->targetDesc = HE::RenderBackendTextureDesc::Create2D(swapChainWidth, swapChainHeight, HE::PixelFormat::BGRA8Unorm, HE::TextureCreateFlags::Present);
	sceneView->targetWidth = swapChainWidth;
	sceneView->targetHeight = swapChainHeight;
	sceneView->camera.aspectRatio = camera.aspectRatio;
	sceneView->camera.fieldOfView = camera.fieldOfView;
	sceneView->camera.zNear = camera.nearPlane;
	sceneView->camera.zFar = camera.farPlane;
	sceneView->camera.position = cameraTransform.position;
	sceneView->camera.euler = cameraTransform.rotation;
	static HE::Quaternion zUpQuat= glm::rotate(glm::quat(), HE::Math::DegreesToRadians(90.0), HE::Vector3(0.0, 1.0, 0.0)) * glm::rotate(glm::quat(), HE::Math::DegreesToRadians(90.0), HE::Vector3(0.0, 0.0, 1.0));
	sceneView->camera.invViewMatrix = HE::Math::Compose(sceneView->camera.position, HE::Quaternion(HE::Math::DegreesToRadians(sceneView->camera.euler)) * zUpQuat, HE::Vector3(1.0f, 1.0f, 1.0f));
	sceneView->camera.viewMatrix = HE::Math::Inverse(sceneView->camera.invViewMatrix);
	sceneView->camera.projectionMatrix = glm::perspectiveRH_ZO(HE::Math::DegreesToRadians(camera.fieldOfView), camera.aspectRatio, camera.nearPlane, camera.farPlane);
	sceneView->camera.invProjectionMatrix = HE::Math::Inverse(sceneView->camera.projectionMatrix);
	RenderSceneView(renderContext, sceneView);

	HE::GRenderGraphResourcePool->Tick();

}

int ModelViewerApp::Run()
{
	while (!IsExitRequest())
	{
		window->ProcessEvents();

		if (window->ShouldClose())
		{
			isExitRequested = true;
		}

		WindowState state = window->GetState();
	
		if (state == WindowState::Minimized)
		{
			continue;
		}

		static std::chrono::steady_clock::time_point previousTimePoint{ std::chrono::steady_clock::now() };
		std::chrono::steady_clock::time_point timePoint = std::chrono::steady_clock::now();
		std::chrono::duration<float> timeDuration = std::chrono::duration_cast<std::chrono::duration<float>>(timePoint - previousTimePoint);
		float deltaTime = timeDuration.count();
		previousTimePoint = timePoint;

		Update(deltaTime);

		uint32 width = window->GetWidth();
		uint32 height = window->GetHeight();
		if (width != swapChainWidth || height != swapChainHeight)
		{
			HE::RenderBackendResizeSwapChain(renderBackend, swapChain, &width, &height);
			swapChainWidth = width;
			swapChainHeight = height;
		}

		Render();

		HE::RenderBackendPresentSwapChain(renderBackend, swapChain);

		((HE::LinearArena*)arena)->Reset();

		frameCounter++;
	}
	return 0;
}

int ModelViewerMain()
{
	int exitCode = EXIT_SUCCESS;
	ModelViewerApp* app = new ModelViewerApp();
	bool result = app->Init();
	if (result)
	{
		exitCode = app->Run();
	}
	else
	{
		exitCode = EXIT_FAILURE;
	}
	app->Exit();
	delete app;
	return exitCode;
}
