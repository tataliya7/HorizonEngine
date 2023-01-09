#pragma once

import HorizonEngine.Core;

struct GLFWwindow;

namespace HE
{
	enum HorizonExampleWindowCreateFlags
	{
		None       = 0,
		Resizable  = 1 << 0,
		Borderless = 1 << 1,
	};

	struct HorizonExampleWindowCreateInfo
	{
		uint32 width;
		uint32 height;
		const char* title;
		HorizonExampleWindowCreateFlags flags;
	};

	enum class HorizonExampleWindowState
	{
		Normal,
		Minimized,
		Maximized,
		Fullscreen,
	};

	class HorizonExampleWindow
	{
	public:

		HorizonExampleWindow(HorizonExampleWindowCreateInfo* info);
		virtual ~HorizonExampleWindow();

		uint32 GetWidth() const
		{
			return width;
		}

		uint32 GetHeight() const
		{
			return height;
		}

		HorizonExampleWindowState GetState() const
		{
			return state;
		}

		uint64 GetNativeHandle();

		GLFWwindow* GetGLFWHandle()
		{
			return handle;
		}

		void ProcessEvents();

		bool ShouldClose() const;

	private:

		GLFWwindow* handle;
		uint32 width;
		uint32 height;
		const char* title;
		bool focused;
		HorizonExampleWindowState state;
	};

	extern bool GLFWInit();
	extern void GLFWExit();
}