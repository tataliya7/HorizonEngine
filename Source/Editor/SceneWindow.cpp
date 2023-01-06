#include "SceneWindow.h"
#include "EditorSceneManager.h"
#include "Editor.h"

#include "Components/Components.h"
#include "Renderer/Renderer.h"

#include "Scene/Camera.h"
#include "Scene/CameraController.h"
#include "Scene/Skeleton.h"

#include <ImGuizmo/ImGuizmo.h>

// TODO
#include "Core/LogManager.h"
#include "Scene/EntityManager.h"
#include "Graphics/Vulkan/VulkanDevice.h"
#include "Graphics/Vulkan/VulkanQueue.h"
#include "Graphics/Vulkan/VulkanBuffer.h"
#include "Graphics/Vulkan/VulkanFramebuffer.h"
#include "RenderingPipelines/tools.h"

namespace Horizon
{
	SceneWindow::SceneWindow(const String& title, Editor* editor)
		: EditorWindow(title, editor)
		, mViewportPos(0, 0)
		, mViewportSize(0, 0)
		, mGizmoOperationType(ImGuizmo::OPERATION::TRANSLATE)
	{

	}

	static void CreateCmdBuffer_Temp(VkCommandBuffer& cmdBuf, VkCommandPool& tmpPool)
	{
		VkCommandPoolCreateInfo commandPoolInfo = {};
		commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		commandPoolInfo.queueFamilyIndex = gDevice->GetGraphicsQueueFamilyIndex();
		commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK(vkCreateCommandPool(gDevice->GetHandle(), &commandPoolInfo, VULKAN_ALLOCATION_CALLBACKS, &tmpPool));

		VkCommandBufferAllocateInfo allocateInfo = {};
		allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocateInfo.commandPool = tmpPool;
		allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocateInfo.commandBufferCount = 1;
		VK_CHECK(vkAllocateCommandBuffers(gDevice->GetHandle(), &allocateInfo, &cmdBuf));

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		VK_CHECK(vkBeginCommandBuffer(cmdBuf, &beginInfo));
	}
	
	static void Flush_Temp(VkCommandBuffer cmdBuf, VkCommandPool pool)
	{
		vkEndCommandBuffer(cmdBuf);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmdBuf;

		auto graphicsQueue = gDevice->GetGraphicsQueue()->GetHandle();
		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

		vkDeviceWaitIdle(gDevice->GetHandle());

		vkDestroyCommandPool(gDevice->GetHandle(), pool, nullptr);
	}

	bool SceneWindow::Pick(MouseButtonPressEvent& e)
	{
		bool handled = false;
		if (e.GetMouseButtonID() == MouseButtonID::ButtonLeft && IsInViewport(e.GetScreenPos()))
		{
			const Vector2& screenPos = e.GetScreenPos();
			Vector2 windowPos = screenPos - mViewportPos;
			mEditor->OnSeleted(mEditor->mSceneContext.sceneRenderer->Pick(windowPos));
			// LOG_DEBUG("mouse pos in scene window: {}, {}. entity handle: {}.", (int32)windowPos.x, (int32)windowPos.y, pickingResult - 1);
		}
		return handled;
	}

	void SceneWindow::OnEvent(Event& e)
	{
		if (ImGuizmo::IsOver())
		{
			return;
		}
		{
			EventDispatcher dispatcher(e);
			dispatcher.Dispatch<MouseButtonPressEvent>(HORIZON_BIND_FUNCTION(SceneWindow::Pick));
		}
		if (e.IsHandled())
		{
			return;
		}
		switch (mEditor->mSceneContext.editorCameraControllerType)
		{
		case EditorCameraControllerType::Orbiter:
		{
			if (!mEditor->mSceneContext.orbiterCameraController->IsControlling())
			{
				EventDispatcher dispatcher(e);
				dispatcher.Dispatch<MouseMoveEvent>(HORIZON_BIND_FUNCTION(SceneWindow::OnMouseMove));
				dispatcher.Dispatch<MouseScrollEvent>(HORIZON_BIND_FUNCTION(SceneWindow::OnMouseScroll));
				dispatcher.Dispatch<MouseButtonPressEvent>(HORIZON_BIND_FUNCTION(SceneWindow::OnMouseButtonPress));
			}
			if (!e.mHandled)
			{
				mEditor->mSceneContext.orbiterCameraController->OnEvent(e);
			}
			break;
		}
		case EditorCameraControllerType::FirstPerson:
		{
			mEditor->mSceneContext.firstPersonCameraController->OnEvent(e);
			break;
		}
		default:
			break;
		}
		
	}

	bool SceneWindow::OnMouseScroll(MouseScrollEvent& e)
	{
		bool handled = false;
		if (!IsInViewport(e.GetScreenPos()))
		{
			handled = true;
		}
		return handled;
	}

	bool SceneWindow::OnMouseMove(MouseMoveEvent& e)
	{
		bool handled = false;
		if (!IsInViewport(e.GetScreenPos()))
		{
			handled = true;
		}
		return handled;
	}

	bool SceneWindow::OnMouseButtonPress(MouseButtonPressEvent& e)
	{
		bool handled = false;
		if (!IsInViewport(e.GetScreenPos()))
		{
			handled = true;
		}
		return handled;
	}

	bool SceneWindow::IsInViewport(const Vector2& pos)
	{
		if ((pos.x > mViewportPos.x) &&
			(pos.x < (mViewportPos.x + mViewportSize.x)) &&
			(pos.y > mViewportPos.y) &&
			(pos.y < (mViewportPos.y + mViewportSize.y)))
		{
			return true;
		}
		return false;
	}

	float SceneWindow::GetSnapValue()
	{
		switch (mGizmoOperationType)
		{
			case  ImGuizmo::OPERATION::TRANSLATE: return 5.0f;
			case  ImGuizmo::OPERATION::ROTATE: return 10.0f;
			case  ImGuizmo::OPERATION::SCALE: return 0.1f;
		}
		return 0.0f;
	}

	void SceneWindow::Draw()
	{
		auto flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_MenuBar;
		ImGui::SetNextWindowBgAlpha(0.0f);

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
		ImGui::Begin(mTitle.c_str(), &mActive, flags);

		mIsFocused = ImGui::IsWindowFocused();

		if(ImGui::BeginMenuBar())
		{
			if (ImGui::Button("T"))
			{
				mGizmoOperationType = ImGuizmo::OPERATION::TRANSLATE;
			}
			else if (ImGui::Button("R"))
			{
				mGizmoOperationType = ImGuizmo::OPERATION::ROTATE;
			}
			else if (ImGui::Button("S"))
			{
				mGizmoOperationType = ImGuizmo::OPERATION::SCALE;
			}
			ImGui::Text("FPS: %.1f (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
			ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
			float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
			ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5f);
			if (ImGui::Button("Editor Camera"))
			{
				ImGui::OpenPopup("Editor Camera");
			}
			if (ImGui::BeginPopup("Editor Camera"))
			{
				static const char* types[] = { "Orbiter", "FirstPerson" };
				int item_type = (int)mEditor->mSceneContext.editorCameraControllerType;
				if (ImGui::Combo("Camera Controller Type", &item_type, types, IM_ARRAYSIZE(types), IM_ARRAYSIZE(types)))
				{
					mEditor->mSceneContext.editorCameraControllerType = (EditorCameraControllerType)item_type;
				}

				PerspectiveCamera* camera = (PerspectiveCamera*)mEditor->mSceneContext.editorCamera.get();
				float yFov = glm::degrees(camera->GetVerticalFieldOfView());
				if (ImGui::DragFloat("Vertical Field Of View", &yFov, 1.0f, 4.0f, 120.0f))
				{
					camera->SetVerticalFieldOfView(glm::radians(yFov));
				}
				float nearPlane = camera->GetNearPlane();
				if (ImGui::DragFloat("Near Plane", &nearPlane))
				{
					camera->SetNearPlane(nearPlane);
				}
				float farPlane = camera->GetFarPlane();
				if (ImGui::DragFloat("Far Plane", &farPlane))
				{
					camera->SetFarPlane(farPlane);
				}
				static float speed = 1.0f;
				if (ImGui::DragFloat("Camera Speed", &speed, 0.01f, 0.001f, 2.0f))
				{
					mEditor->mSceneContext.orbiterCameraController->mSpeed = speed;
					mEditor->mSceneContext.firstPersonCameraController->mSpeed = speed;
				}
				ImGui::EndPopup();
			}
			ImGui::EndMenuBar();
		}
		
		auto viewportSize = ImGui::GetContentRegionAvail();
		mViewportSize = Vector2(viewportSize.x, viewportSize.y);

		if (mEditor->mSceneContext.sceneImageHandle != nullptr)
		{
			ImGui::Image(mEditor->mSceneContext.sceneImageHandle, viewportSize, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f));
		}

		float windowWidth = ImGui::GetWindowWidth();
		float windowHeight = ImGui::GetWindowHeight();
		float delta = windowHeight - viewportSize.y;
		mViewportPos = Vector2(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y + delta);

		// Gizmos
		Entity* selectedEntity = mEditor->mSceneContext.selectedEntity;
		Bone* selectedBone = mEditor->mSceneContext.selectedBone;
		if (selectedEntity && mGizmoOperationType != -1)
		{
			ImGuizmo::SetOrthographic(false);
			ImGuizmo::SetDrawlist();

			ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y + delta, viewportSize.x, viewportSize.y);

			// Editor camera
			Matrix4 cameraProjection = mEditor->mSceneContext.editorCamera->GetProjectionMatrix();
			cameraProjection[1][1] *= -1;
			Matrix4 cameraView = mEditor->mSceneContext.editorCamera->GetViewMatrix();

			static Matrix4 transformMatrix = Matrix4(1.0f);

			// Entity transform
			auto& transformComponent = selectedEntity->GetComponent<TransformComponent>();

			if (selectedBone == nullptr)
			{
				transformMatrix = transformComponent.localToWorldMatrix;
			}
			else
			{
				transformMatrix = selectedBone->GetLocalMatrix();
			}

			bool snap = Input::IsKeyPressed(KeyCode::LeftControl);

			float snapValue = GetSnapValue();
			float snapValues[3] = { snapValue, snapValue, snapValue };

			float deltaMatrix[16];
			ImGuizmo::Manipulate(glm::value_ptr(cameraView), 
				glm::value_ptr(cameraProjection),
				(ImGuizmo::OPERATION)mGizmoOperationType,
				ImGuizmo::LOCAL, 
				glm::value_ptr(transformMatrix),
				nullptr,
				snap ? snapValues : nullptr);

			ImGuizmo::DrawGrid(glm::value_ptr(cameraView),
				glm::value_ptr(cameraProjection),
				glm::value_ptr(Matrix4(1)),
				100.0f);

			if (ImGuizmo::IsUsing())
			{
				auto parent = selectedEntity->GetCreator()->GetEntityByHandle(selectedEntity->GetComponent<SceneHierarchyComponent>().parent);
				if (parent)
				{
					transformMatrix = glm::inverse(parent->GetComponent<TransformComponent>().localToWorldMatrix) * transformMatrix;
				}
				selectedEntity->ReplaceComponent<TransformComponent>(transformMatrix);
			}
		}

		ImGui::End();
		ImGui::PopStyleVar();
	}
}
