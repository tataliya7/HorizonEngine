#include "SceneHierarchyWindow.h"
#include "EditorSceneManager.h"
#include "Editor.h"

#include "Scene/EntityManager.h"
#include "Scene/Entity.h"
#include "Scene/Scene.h"

namespace Horizon
{
	SceneHierarchyWindow::SceneHierarchyWindow(const String& title, Editor* editor)
		: EditorWindow(title, editor)
	{

	}
	
	SceneHierarchyWindow::~SceneHierarchyWindow()
	{

	}

	void SceneHierarchyWindow::DrawEntityNode_Internal(Entity* entity, EntityManager* entityManager)
	{
		auto& entityName = entity->mName;

		const auto& hierarchy = entity->GetComponent<SceneHierarchyComponent>();
		ImGuiTreeNodeFlags flags = ((mEditor->mSceneContext.selectedEntity != nullptr && mEditor->mSceneContext.selectedEntity == entity) ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
		if (hierarchy.numChildren == 0)
		{
			flags |= ImGuiTreeNodeFlags_Leaf;
		}
		bool opened = ImGui::TreeNodeEx((void*)(uint64)entity->GetHandle(), flags, entityName.c_str());
		if (ImGui::IsItemClicked())
		{
			mEditor->mSceneContext.selectedBone = nullptr;
			mEditor->mSceneContext.selectedEntity = entity;
		}

		bool entityDeleted = false;
		if (ImGui::BeginPopupContextItem())
		{
			if (ImGui::MenuItem("Delete Entity"))
			{
				entityDeleted = true;
			}

			ImGui::EndPopup();
		}

		if (opened)
		{
			auto curr = hierarchy.first;
			for (uint32 i = 0; i < hierarchy.numChildren; i++)
			{
				DrawEntityNode_Internal(entityManager->GetEntityByHandle(curr), entityManager);
				curr = entityManager->GetEntityByHandle(curr)->GetComponent<SceneHierarchyComponent>().next;
			}
			ImGui::TreePop();
		}

		if (entityDeleted)
		{
			entity->GetCreator()->DestroyEntity(entity->GetHandle());
			if (mEditor->mSceneContext.selectedEntity == entity)
			{
				mEditor->mSceneContext.selectedEntity = nullptr;
			}
		}
	}

	void SceneHierarchyWindow::Draw()
	{
		ImGui::Begin(mTitle.c_str(), &mActive);

		ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_DefaultOpen;
		if (ImGui::TreeNodeEx((void*)(uint64)564788, flags, "Sample Scene"))
		{
			auto entityManager = EditorSceneManager::GetActiveScene()->GetEntityManager();
			entityManager->GetRegistry()->each([&](auto entityHandle)
			{
				auto entity = entityManager->GetEntityByHandle(entityHandle);
				if (entity->GetComponent<SceneHierarchyComponent>().parent == ENTITY_NULL_HANDLE)
				{
					DrawEntityNode_Internal(entity, entityManager);
				}
			});
			ImGui::TreePop();
		}

		if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered())
		{
			mEditor->mSceneContext.selectedEntity = nullptr;
		}

		// Right-click on blank space
		if (ImGui::BeginPopupContextWindow(0, 1, false))
		{
			if (ImGui::MenuItem("Create Empty Entity"))
			{
				EditorSceneManager::GetActiveScene()->GetEntityManager()->CreateEntity("Empty Entity");
			}

			ImGui::EndPopup();
		}
		ImGui::End();
	}
}
