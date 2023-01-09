#include "HorizonExampleWindow.h"

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <windows.h>

#include "Core/CoreDefinitions.h"

namespace HE
{
	static void ErrorCallback(int errorCode, const char* description)
	{
		HE_LOG_ERROR("GLFW error occurs. [error code]: {}, [desctiption]: {}.", errorCode, description);
	}

	bool GLFWInit()
	{
		glfwSetErrorCallback(ErrorCallback);
		if (glfwInit() != GLFW_TRUE)
		{
			HE_LOG_ERROR("Failed to init glfw.");
			return false;
		}
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		return true;
	}

	void GLFWExit()
	{
		glfwTerminate();
	}

	HorizonExampleWindow::HorizonExampleWindow(HorizonExampleWindowCreateInfo* info)
	{
		handle = glfwCreateWindow(info->width, info->height, info->title, nullptr, nullptr);
		if (!handle)
		{
			HE_LOG_FATAL("Failed to create main window");
			return;
		}

		if (info->flags & HorizonExampleWindowCreateFlags::Resizable)
		{
			glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
		}
		else
		{
			glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		}

		if (info->flags & HorizonExampleWindowCreateFlags::Borderless)
		{
			glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
		}
		else
		{
			glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
		}

		title = info->title;

		int32 w, h;
		glfwGetWindowSize(handle, &w, &h);

		width = w;
		height = h;

		glfwShowWindow(handle);
		glfwFocusWindow(handle);
		focused = true;

		glfwSetWindowUserPointer(handle, this);
		glfwSetWindowFocusCallback(handle, [](GLFWwindow* glfwWindow, int focused)
		{
			HorizonExampleWindow* window = (HorizonExampleWindow*)glfwGetWindowUserPointer(glfwWindow);
			window->focused = (focused == GLFW_TRUE) ? true : false;
		});
		glfwSetWindowSizeCallback(handle, [](GLFWwindow* glfwWindow, int width, int height)
		{
			HorizonExampleWindow* window = (HorizonExampleWindow*)glfwGetWindowUserPointer(glfwWindow);
			glfwSetWindowSize(glfwWindow, width, height);
			int32 w, h;
			glfwGetWindowSize(glfwWindow, &w, &h);
			window->width = w;
			window->height = h;
			if (window->width == 0 || window->height == 0)
			{
				window->state = HorizonExampleWindowState::Minimized;
			}
			else if (glfwGetWindowMonitor(glfwWindow) != nullptr)
			{
				window->state = HorizonExampleWindowState::Fullscreen;
			}
			else
			{
				window->state = HorizonExampleWindowState::Normal;
			}
		});
		glfwSetWindowMaximizeCallback(handle, [](GLFWwindow* glfwWindow, int maximized)
		{
			HorizonExampleWindow* window = (HorizonExampleWindow*)glfwGetWindowUserPointer(glfwWindow);
			if (maximized == GLFW_TRUE)
			{
				window->state = HorizonExampleWindowState::Maximized;
			}
		});
		glfwSetWindowCloseCallback(handle, [](GLFWwindow* glfwWindow)
		{
			glfwSetWindowShouldClose(glfwWindow, true);
		});
	}

	HorizonExampleWindow::~HorizonExampleWindow()
	{
		if (handle)
		{
			glfwDestroyWindow(handle);
		}
	}

	uint64 HorizonExampleWindow::GetNativeHandle()
	{
		return (uint64)glfwGetWin32Window(handle);
	}

	void HorizonExampleWindow::ProcessEvents()
	{
		glfwPollEvents();
	}

	bool HorizonExampleWindow::ShouldClose() const
	{
		return glfwWindowShouldClose(handle);
	}
}