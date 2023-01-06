#pragma once

#include "CameraController.h"
#include "ECS/ECS.h"
#include "HybridRenderPipeline/HybridRenderPipeline.h"
#include "DxcShaderCompiler/DxcShaderCompiler.h"

#define HE_APPLICATION_NAME "Model Viewer"
#define HE_APPLICATION_VERSION HE_MAKE_VERSION(1, 0, 0)

import HorizonEngine.SceneManagement;

namespace HE
{
	class MainWindow;
	class Application
	{
	public:

		static Application* Instance;

		static Application* GetInstance()
		{
			return Instance;
		}

		Application();
		~Application();

		Application(const Application&) = delete;
		Application& operator=(const Application&) = delete;

		bool Init();
		void Exit();


		int Run();

		float CalculateDeltaTime();
		void Update(float deltaTime);
		void Render();
		void OnImGui();
		void DrawEntityNodeUI(EntityHandle entity);
		
		bool IsExitRequest() const
		{
			return isExitRequested;
		}

		MainWindow* GetMainWindow()
		{
			return window;
		}

	private:

		MemoryArena* arena;

		const char* name;

		bool isExitRequested;

		uint64 frameCounter;

		MainWindow* window;

		Scene* activeScene;

		RenderScene* renderScene;

		EntityHandle mainCamera;
		EntityHandle sky;

		RenderBackendSwapChainHandle swapChain;
		uint32 swapChainWidth;
		uint32 swapChainHeight;

		ShaderCompiler* shaderCompiler;
		RenderBackend* renderBackend;
		UIRenderer* uiRenderer;

		EntityHandle selectedEntity;

		RenderContext* renderContext;

		SceneView* sceneView;
		HybridRenderPipeline* renderPipeline;

	};

	extern int ApplicationMain();
}