#pragma once

#include "Core/CoreDefinitions.h"
#include "HorizonExampleWindow.h"

#include <string>
#include <optick.h>

import HorizonEngine.Core;
import HorizonEngine.Entity;
import HorizonEngine.Physics;
import HorizonEngine.Render;
import HorizonEngine.Audio;
import HorizonEngine.Scene;
import HorizonEngine.Daisy;
import HorizonEngine.Render.VulkanRenderBackend;

namespace HE
{
	class HorizonExampleWindow;
	class DaisyRenderer;

	class HorizonExampleBase
	{
	public:

		static HorizonExampleBase* Instance;
		static HorizonExampleBase* GetInstance()
		{
			return Instance;
		}

		HorizonExampleBase();
		virtual ~HorizonExampleBase();

		HorizonExampleBase(HorizonExampleBase&) = delete;
		HorizonExampleBase(const HorizonExampleBase&) = delete;
		HorizonExampleBase& operator=(HorizonExampleBase&) = delete;
		HorizonExampleBase& operator=(const HorizonExampleBase&) = delete;

		bool Init();
		void Exit();
		int Run();

		void Tick();

		virtual void Setup() = 0;
		virtual void Clear() = 0;

		virtual void OnUpdate(float deltaTime) = 0;
		virtual void OnRender() = 0;
		virtual void OnDrawUI() = 0;

		float CalculateDeltaTime();

		uint64 GetFrameCounter() const
		{
			return frameCounter;
		}

		bool IsExitRequest() const
		{
			return isExitRequested;
		}

		void SetExitRequest(bool value)
		{
			isExitRequested = value;
		}

		HorizonExampleWindow* GetMainWindow()
		{
			return window;
		}

	protected:

		std::string name = "Horizon Example";

		uint32 initialWidth = 1920;
		uint32 initialHeight = 1080;

		bool showOverlay = true;

		DaisyRenderer* daisyRenderer = nullptr;
		RenderBackend* renderBackend = nullptr;

		RenderBackendSwapChainHandle swapChain;
		uint32 swapChainWidth;
		uint32 swapChainHeight;

	private:

		uint64 frameCounter = 0;
		bool isExitRequested = false;
	
		HorizonExampleWindow* window = nullptr;
	};
}

#define HORIZON_EXAMPLE_MAIN()                         \
int HorizonExampleMain()                               \
{                                                      \
    using namespace HE;                                \
	int exitCode = EXIT_SUCCESS;                       \
	HorizonExample* example = new HorizonExample();    \
	bool result = example->Init();                     \
	if (result)                                        \
	{                                                  \
		exitCode = example->Run();                     \
	}                                                  \
	else                                               \
	{                                                  \
		exitCode = EXIT_FAILURE;                       \
	}                                                  \
	example->Exit();                                   \
	delete example;                                    \
	return exitCode;                                   \
}