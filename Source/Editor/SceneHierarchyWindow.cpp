#include "SceneHierarchyWindow.h"

#include "ECS/ECS.h"

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

import HorizonEngine.SceneManagement;

namespace HE
{
	SceneHierarchyWindow::SceneHierarchyWindow(const String& title, Editor* editor)
		: EditorWindow(title, editor)
	{

	}
	
	SceneHierarchyWindow::~SceneHierarchyWindow()
	{

	}

	template<typename ComponentType>
	void DrawComponent(const char* lable, ComponentType& component)
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

	void DrawEntityCreateMenu()
	{
		
	}

	void SceneHierarchyWindow::DrawEntityNode(EntityHandle entity)
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

	void SceneHierarchyWindow::OnDrawUI()
	{
		ImGui::Begin("Scene Hierarchy", &open);

		focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

		ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_DefaultOpen;

		if (ImGui::TreeNodeEx((void*)(uint64)564788, flags, "Sample Scene"))
		{
			auto entityManager = SceneManager::GetActiveScene()->GetEntityManager();
			for (const auto& entity : entityManager->GetView<SceneHierarchyComponent>())
			{
				if (entityManager->GetComponent<SceneHierarchyComponent>(entity).parent == entt::null)
				{
					DrawEntityNode(entity, entityManager);
				}
			});
			ImGui::TreePop();
		}

		if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered())
		{
			mEditor->mSceneContext.selectedEntity = nullptr;
		}

		if (ImGui::BeginPopupContextWindow(nullptr, ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems))
		{
			if (ImGui::BeginMenu("Create"))
			{
				if (ImGui::MenuItem("Empty Entity"))
				{
					EditorSceneManager::GetActiveScene()->GetEntityManager()->CreateEntity("Empty Entity");

					SelectionManager::DeselectAll();
					SelectionManager::Select(s_ActiveSelectionContext, newEntity.GetUUID());
				}
			}
			ImGui::EndPopup();
		}
		
		ImGui::End();
	}
}
