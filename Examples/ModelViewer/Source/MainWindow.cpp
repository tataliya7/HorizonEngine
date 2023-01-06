#include "MainWindow.h"

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

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

MainWindow::MainWindow(WindowCreateInfo* info)
{
	handle = glfwCreateWindow(info->width, info->height, info->title, nullptr, nullptr);
	if (!handle)
	{
		HE_LOG_FATAL("Failed to create main window");
		return;
	}

	if (HAS_ANY_FLAGS(info->flags, WindowFlags::Resizable))
	{
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
	}
	else
	{
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	}

	if (HAS_ANY_FLAGS(info->flags, WindowFlags::Borderless))
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
	glfwSetWindowFocusCallback(handle, [](GLFWwindow* window, int focused)
	{ 
		MainWindow* mainWindow = (MainWindow*)glfwGetWindowUserPointer(window);
		mainWindow->focused = (focused == GLFW_TRUE) ? true : false;
	});
	glfwSetWindowSizeCallback(handle, [](GLFWwindow* window, int width, int height)
	{	
		MainWindow* mainWindow = (MainWindow*)glfwGetWindowUserPointer(window);
		glfwSetWindowSize(window, width, height);
		int32 w, h;
		glfwGetWindowSize(window, &w, &h);
		mainWindow->width = w;
		mainWindow->height = h;
		if (mainWindow->width == 0 || mainWindow->height == 0)
		{
			mainWindow->state = WindowState::Minimized;
		}
		else if (glfwGetWindowMonitor(window) != nullptr)
		{
			mainWindow->state = WindowState::Fullscreen;
		}
		else
		{
			mainWindow->state = WindowState::Normal;
		}
	});
	glfwSetWindowMaximizeCallback(handle, [](GLFWwindow* window, int maximized)
	{
		MainWindow* mainWindow = (MainWindow*)glfwGetWindowUserPointer(window);
		if (maximized == GLFW_TRUE)
		{
			mainWindow->state = WindowState::Maximized;
		}
	});
	glfwSetWindowCloseCallback(handle, [](GLFWwindow* window)
	{
		glfwSetWindowShouldClose(window, true);
	});
}

MainWindow::~MainWindow()
{
	if (handle)
	{
		glfwDestroyWindow(handle);
	}
}

HWND MainWindow::GetNativeHandle()
{
	return glfwGetWin32Window(handle);
}

void MainWindow::ProcessEvents()
{
	glfwPollEvents();
}

bool MainWindow::ShouldClose() const
{ 
	return glfwWindowShouldClose(handle);
}