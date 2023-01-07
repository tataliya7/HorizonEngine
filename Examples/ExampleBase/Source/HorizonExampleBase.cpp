#include <chrono>

#include "HorizonExampleBase.h"
#include "HorizonExampleWindow.h"
#include "Daisy/DaisyRenderer.h"

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

import HorizonEngine.Render.VulkanRenderBackend;
import HorizonEngine.Input;
import HorizonEngine.Audio;
import HorizonEngine.Script;

namespace HE
{
	void DrawOverlay()
	{
		static int corner = 0;
		ImGuiIO& io = ImGui::GetIO();
		ImGuiWindowFlags windowFags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
		if (corner != -1)
		{
			const float padding = 10.0f;
			const ImGuiViewport* viewport = ImGui::GetMainViewport();
			ImVec2 workPos = viewport->WorkPos; // Use work area to avoid menu-bar/task-bar, if any!
			ImVec2 workSize = viewport->WorkSize;
			ImVec2 windowPos, windowPosPivot;
			windowPos.x = (corner & 1) ? (workPos.x + workSize.x - padding) : (workPos.x + padding);
			windowPos.y = (corner & 2) ? (workPos.y + workSize.y - padding) : (workPos.y + padding);
			windowPosPivot.x = (corner & 1) ? 1.0f : 0.0f;
			windowPosPivot.y = (corner & 2) ? 1.0f : 0.0f;
			ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always, windowPosPivot);
			ImGui::SetNextWindowViewport(viewport->ID);
			windowFags |= ImGuiWindowFlags_NoMove;
		}
		ImGui::SetNextWindowBgAlpha(0.35f); // Transparent background
		static bool open = true;
		if (ImGui::Begin("Overlay", &open, windowFags))
		{
			ImGui::Text("Horizon Engine");
			ImGui::Separator();
			ImGui::Text("FPS: %.1f (%.2f ms/frame)", ImGui::GetIO().Framerate, (1000.0f / ImGui::GetIO().Framerate));
		}
		ImGui::End();
	}
}

namespace HE
{
	HorizonExampleBase* HorizonExampleBase::Instance = nullptr;

	HorizonExampleBase::HorizonExampleBase()
	{
		ASSERT(!Instance);
		Instance = this;
	}

	HorizonExampleBase::~HorizonExampleBase()
	{
		ASSERT(Instance);
		Instance = nullptr;
	}

	bool HorizonExampleBase::Init()
	{
		GLFWInit();
		PhysXInit();
		AudioEngine::Init();

		HorizonExampleWindowCreateInfo windowInfo = {
			.width = initialWidth,
			.height = initialHeight,
			.title = name.c_str(),
			.flags = HorizonExampleWindowCreateFlags::Resizable
		};
		window = new HorizonExampleWindow(&windowInfo);

		Input::SetCurrentContext(window->GetGLFWHandle());

		daisyRenderer = new DaisyRenderer(window->GetGLFWHandle());
		renderBackend = daisyRenderer->GetRenderBackend();
		swapChain = RenderBackendCreateSwapChain(renderBackend, daisyRenderer->GetPrimaryDeviceMask(), (uint64)window->GetNativeHandle());
		swapChainWidth = window->GetWidth();
		swapChainHeight = window->GetHeight();

		Setup();

		return true;
	}

	void HorizonExampleBase::Exit()
	{
		Clear();

		delete daisyRenderer;
		delete window;

		AudioEngine::Exit();
		PhysXExit();
		GLFWExit();
		LogSystemExit();
	}

	float HorizonExampleBase::CalculateDeltaTime()
	{
		static std::chrono::steady_clock::time_point previousTimePoint{ std::chrono::steady_clock::now() };
		std::chrono::steady_clock::time_point timePoint = std::chrono::steady_clock::now();
		std::chrono::duration<float> timeDuration = std::chrono::duration_cast<std::chrono::duration<float>>(timePoint - previousTimePoint);
		float deltaTime = timeDuration.count();
		previousTimePoint = timePoint;
		return deltaTime;
	}

	void HorizonExampleBase::Update()
	{
		OPTICK_EVENT();
		
		float deltaTime = CalculateDeltaTime();
		OnUpdate(deltaTime);
	}

	void HorizonExampleBase::Render()
	{
		OPTICK_EVENT();

		OnRender();
	}

	void HorizonExampleBase::DrawUI()
	{
		OPTICK_EVENT();

		OnDrawUI();

		if (showOverlay)
		{
			DrawOverlay();
		}
	}

	int HorizonExampleBase::Run()
	{
		while (!IsExitRequest())
		{
			OPTICK_FRAME("MainThread");

			window->ProcessEvents();

			if (window->ShouldClose())
			{
				SetExitRequest(true);
			}

			HorizonExampleWindowState state = window->GetState();

			if (state == HorizonExampleWindowState::Minimized)
			{
				continue;
			}

			uint32 width = window->GetWidth();
			uint32 height = window->GetHeight();
			if (width != swapChainWidth || height != swapChainHeight)
			{
				RenderBackendResizeSwapChain(renderBackend, swapChain, &width, &height);
				swapChainWidth = width;
				swapChainHeight = height;
			}

			Update();

			daisyRenderer->BeginFrame();

			DrawUI();

			Render();

			daisyRenderer->EndFrame();

			RenderBackendPresentSwapChain(renderBackend, swapChain);

			frameCounter++;
		}
		return 0;
	}
}