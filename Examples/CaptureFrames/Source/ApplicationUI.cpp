#include "Application.h"
#include "HybridRenderPipeline/HybridRenderPipeline.h"

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

namespace HE
{
	std::unordered_map<std::string, std::function<void(const char*, const char* name, void*)>> uiCreator;
	std::vector<std::pair<std::string, bool>> g_editor_node_state_array;
	int                                       g_node_depth = -1;
	bool inited = false;

	void UIInit()
	{
		using namespace HE;

		uiCreator["bool"] = [](const char* lable, const char* name, void* value)
		{
			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted(name);
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			ImGui::Checkbox(lable, static_cast<bool*>(value));
			ImGui::PopItemWidth();
			ImGui::NextColumn();
		};

		uiCreator["int"] = [](const char* lable, const char* name, void* value)
		{
			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted(name);
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			ImGui::DragInt(lable, static_cast<int*>(value));
			ImGui::PopItemWidth();
			ImGui::NextColumn();
		};

		uiCreator["unsigned int"] = [](const char* lable, const char* name, void* value)
		{
			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted(name);
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			ImGui::DragInt(lable, static_cast<int*>(value));
			ImGui::PopItemWidth();
			ImGui::NextColumn();
		};

		uiCreator["float"] = [](const char* lable, const char* name, void* value)
		{
			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted(name);
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			ImGui::DragFloat(lable, static_cast<float*>(value));
			ImGui::PopItemWidth();
			ImGui::NextColumn();
		};

		uiCreator["struct glm::vec<3,float,0>"] = [](const char* lable, const char* name, void* value)
		{
			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted(name);
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			ImGui::DragFloat3(lable, static_cast<float*>(value));
			ImGui::PopItemWidth();
			ImGui::NextColumn();
		};

		uiCreator["class std::basic_string<char,struct std::char_traits<char>,class std::allocator<char> >"] = [](const char* lable, const char* name, void* value)
		{
			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted(name);
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			ImGui::Text(lable, static_cast<std::string*>(value)->c_str());
			/*if (ImGui::InputText(lable, static_cast<std::string*>(value)->c_str(), 256))
			{
				
			}*/
			ImGui::PopItemWidth();
			ImGui::NextColumn();
		};
	}

	void BeginDockSpace()
	{
		static bool dockSpaceOpen = true;

		// Imgui dock node flags.
		static ImGuiDockNodeFlags dockNodeflags = ImGuiDockNodeFlags_PassthruCentralNode;

		// Imgui window flags.
		ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDocking;

		static bool isFullscreenPersistant = true;
		bool isFullscreen = isFullscreenPersistant;
		if (isFullscreen)
		{
			ImGuiViewport* viewport = ImGui::GetMainViewport();
			ImGui::SetNextWindowPos(viewport->Pos);
			ImGui::SetNextWindowSize(viewport->Size);
			ImGui::SetNextWindowViewport(viewport->ID);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
			windowFlags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
			windowFlags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
		}

		windowFlags |= ImGuiWindowFlags_NoBackground;

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		// When using ImGuiDockNodeFlags_PassthruDockspace, DockSpace() will render our background and handle the pass-thru hole, so we ask Begin() to not render a background.
		ImGui::Begin("Dockspace", &dockSpaceOpen, windowFlags);

		ImGui::PopStyleVar();

		if (isFullscreen)
		{
			ImGui::PopStyleVar(2);
		}

		// Set min width
		ImGuiIO& io = ImGui::GetIO();
		ImGuiStyle& style = ImGui::GetStyle();
		float minWinSizeX = style.WindowMinSize.x;
		style.WindowMinSize.x = 300.0f;
		if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
		{
			ImGuiID dockSpaceID = ImGui::GetID("MyDockSpace");
			ImGui::DockSpace(dockSpaceID, ImVec2(0.0f, 0.0f), dockNodeflags);
		}
		style.WindowMinSize.x = minWinSizeX;
	}

	void EndDockSpace()
	{
		ImGui::End();
	}

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

	void DrawProfilerWindow()
	{
		static bool showProfilerWindow = true;
		if (ImGui::Begin("Profiler", &showProfilerWindow))
		{
			ImGui::Text("FPS: %.1f (%.2f ms/frame)", ImGui::GetIO().Framerate, (1000.0f / ImGui::GetIO().Framerate));

			if (ImGui::CollapsingHeader("CPU Time", ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::Text("CPU Frametime: %.2f ms", 1000.0);
			}

			if (ImGui::CollapsingHeader("GPU Time", ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::Text("GPU Frametime: %.2f ms", 0.0f);

				//for (uint32 i = 0; i < (uint32)sceneRenderer->timings.size(); i++)
				//ImGui::Text("%-30s: %.4f ms", sceneRenderer->timings[i].label.c_str(), sceneRenderer->timings[i].time);
			}
		}
		ImGui::End();
	}

	template<typename ComponentType>
	void DrawComponentUI(const char* lable, ComponentType& component)
	{
		using namespace entt;
		if (ImGui::CollapsingHeader(lable, ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
			ImGui::Columns(2);
			ImGui::Separator();
			for (auto data : entt::resolve<ComponentType>().data())
			{
				std::string type = std::string(data.type().info().name());
				std::string name = data.prop("Name"_hs).value().cast<std::string>();
				void* value = data.get(component).data();
				uiCreator[type](std::string("##" + name).c_str(), name.c_str(), value);
			}
			ImGui::Columns(1);
			ImGui::Separator();
			ImGui::PopStyleVar();
		}
	}

	void Application::DrawEntityNodeUI(EntityHandle entity)
	{
		auto entityManager = SceneManager::GetActiveScene()->GetEntityManager();
		const auto& hierarchy = entityManager->GetComponent<SceneHierarchyComponent>(entity);
		ImGuiTreeNodeFlags flags = ((selectedEntity != EntityHandle() && selectedEntity == entity) ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
		if (hierarchy.numChildren == 0)
		{
			flags |= ImGuiTreeNodeFlags_Leaf;
		}
		bool opened = ImGui::TreeNodeEx((void*)(uint64)entity, flags, entityManager->GetComponent<NameComponent>(entity).name.c_str());
		if (ImGui::IsItemClicked())
		{
			selectedEntity = entity;
		}

		if (opened)
		{
			auto currentEntity = hierarchy.first;
			for (uint32 i = 0; i < hierarchy.numChildren; i++)
			{
				DrawEntityNodeUI(currentEntity);
				currentEntity = entityManager->GetComponent<SceneHierarchyComponent>(currentEntity).next;
			}
			ImGui::TreePop();
		}

		bool deleted = false;
		if (ImGui::BeginPopupContextItem())
		{
			if (ImGui::MenuItem("Delete Entity"))
			{
				deleted = true;
			}
			ImGui::EndPopup();
		}

		if (deleted)
		{
			entityManager->DestroyEntity(entity);
			if (selectedEntity == entity)
			{
				selectedEntity = EntityHandle();
			}
		}
	}

	void Application::OnImGui()
	{
		using namespace HE;
		using namespace entt;

		BeginDockSpace();

		if (!inited)
		{
			UIInit();
			inited = true;
		}

		static bool showSceneHierarchyWindow = true;
		if (ImGui::Begin("Scene Hierarchy", &showSceneHierarchyWindow))
		{
			ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_DefaultOpen;
			if (ImGui::TreeNodeEx((void*)(uint64)564788, flags, SceneManager::GetActiveScene()->name.c_str()))
			{
				auto entityManager = SceneManager::GetActiveScene()->GetEntityManager();
				entityManager->Get()->each([&](auto entity)
				{
					if (entityManager->GetComponent<SceneHierarchyComponent>(entity).parent == EntityHandle())
					{
						DrawEntityNodeUI(entity);
					}
				});
				ImGui::TreePop();
			}

			if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered())
			{
				selectedEntity = {};
			}

			// Right-click on blank space
			if (ImGui::BeginPopupContextWindow(0, 1, false))
			{
				if (ImGui::MenuItem("Create Empty Entity"))
				{
					auto newEntity = SceneManager::GetActiveScene()->GetEntityManager()->CreateEntity("Empty Entity");
					SceneManager::GetActiveScene()->GetEntityManager()->AddComponent<TransformComponent>(newEntity);
					SceneManager::GetActiveScene()->GetEntityManager()->AddComponent<SceneHierarchyComponent>(newEntity);
				}
				ImGui::EndPopup();
			}
		}
		ImGui::End();

		static bool showInspectorWindow = true;
		if (ImGui::Begin("Inspector", &showInspectorWindow))
		{
			if (selectedEntity != EntityHandle())
			{
				ImGui::PushID((int)uint64(selectedEntity));

				auto& transformComponent = activeScene->GetEntityManager()->GetComponent<TransformComponent>(selectedEntity);
				if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
					ImGui::Columns(2);
					ImGui::Separator();

					bool dirty = false;
					ImGui::AlignTextToFramePadding();
					ImGui::TextUnformatted("Position");
					ImGui::NextColumn();
					ImGui::PushItemWidth(-1);
					if (ImGui::DragFloat3("##Position", static_cast<float*>(&transformComponent.position.x)))
					{
						dirty = true;
					}
					ImGui::PopItemWidth();
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding();
					ImGui::TextUnformatted("Rotation");
					ImGui::NextColumn();
					ImGui::PushItemWidth(-1);
					if (ImGui::DragFloat3("##Rotation", static_cast<float*>(&transformComponent.rotation.x)))
					{
						dirty = true;
					}
					ImGui::PopItemWidth();
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding();
					ImGui::TextUnformatted("Scale");
					ImGui::NextColumn();
					ImGui::PushItemWidth(-1);
					if (ImGui::DragFloat3("##Scale", static_cast<float*>(&transformComponent.scale.x)))
					{
						dirty = true;
					}
					ImGui::PopItemWidth();
					ImGui::NextColumn();

					ImGui::Columns(1);
					ImGui::Separator();
					ImGui::PopStyleVar();

					if (dirty)
					{
						activeScene->GetEntityManager()->ReplaceComponent<TransformComponent>(selectedEntity, transformComponent);
					}
				}

				if (auto* component = activeScene->GetEntityManager()->TryGetComponent<DirectionalLightComponent>(selectedEntity))
				{
					DrawComponentUI<DirectionalLightComponent>("Directional Light", *component);
				}

				if (auto* component = activeScene->GetEntityManager()->TryGetComponent<SkyLightComponent>(selectedEntity))
				{
					DrawComponentUI<SkyLightComponent>("Sky Light", *component);
				}

				if (auto* component = activeScene->GetEntityManager()->TryGetComponent<CameraComponent>(selectedEntity))
				{
					DrawComponentUI<CameraComponent>("Camera", *component);
				}

				if (auto* component = activeScene->GetEntityManager()->TryGetComponent<StaticMeshComponent>(selectedEntity))
				{
					DrawComponentUI<StaticMeshComponent>("Static Mesh", *component);
				}
				ImGui::PopID();
			}
		}
		ImGui::End();

		DrawOverlay();

		DrawProfilerWindow();

		EndDockSpace();
	}
}