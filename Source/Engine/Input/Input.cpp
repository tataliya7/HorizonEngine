module;

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

module HorizonEngine.Input;

namespace HE::Input
{
	GLFWwindow* gCurrentWindow = nullptr;

	void SetCurrentContext(void* window)
	{
		gCurrentWindow = (GLFWwindow*)window;
	}

	bool GetKeyDown(KeyCode key)
	{
		int state = glfwGetKey(gCurrentWindow, (int)key);
		return state == GLFW_PRESS;
	}

	bool GetKeyUp(KeyCode key)
	{
		int state = glfwGetKey(gCurrentWindow, (int)key);
		return state == GLFW_RELEASE;
	}

	bool GetMouseButtonDown(MouseButtonID button)
	{
		int state = glfwGetMouseButton(gCurrentWindow, (int)button);
		return state == GLFW_PRESS;
	}

	bool GetMouseButtonUp(MouseButtonID button)
	{
		int state = glfwGetMouseButton(gCurrentWindow, (int)button);
		return state == GLFW_RELEASE;
	}

	void GetMousePosition(float& x, float& y)
	{
		double xpos, ypos;
		glfwGetCursorPos(gCurrentWindow, &xpos, &ypos);
		x = (float)xpos;
		y = (float)ypos;
	}
}
