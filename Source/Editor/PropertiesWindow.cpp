#include "InspectorWindow.h"
#include "Editor.h"

#include "Math/MathUtil.h"
#include "Scene/Mesh.h"
#include "Scene/Light.h"
#include "Scene/Material.h"
#include "Scene/Camera.h"
#include "Scene/Entity.h"
#include "Scene/EntityManager.h"
#include "Components/Components.h"

namespace Horizon
{
	InspectorWindow::InspectorWindow(const String& title, Editor* editor)
		: EditorWindow(title, editor)
	{
		
	}

	static void DrawVec3Control(const String& label, Vector3& values, float resetValue = 0.0f, float columnWidth = 100.0f)
	{
		ImGuiIO& io = ImGui::GetIO();
		auto boldFont = io.Fonts->Fonts[0];

		ImGui::PushID(label.c_str());

		ImGui::Columns(2);
		ImGui::SetColumnWidth(0, columnWidth);
		ImGui::Text(label.c_str());
		ImGui::NextColumn();

		ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{ 0, 0 });

		float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		ImVec2 buttonSize = { lineHeight + 3.0f, lineHeight };

		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f });
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f });
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f });
		ImGui::PushFont(boldFont);
		if (ImGui::Button("X", buttonSize))
			values.x = resetValue;
		ImGui::PopFont();
		ImGui::PopStyleColor(3);

		ImGui::SameLine();
		ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.2f");
		ImGui::PopItemWidth();
		ImGui::SameLine();

		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
		ImGui::PushFont(boldFont);
		if (ImGui::Button("Y", buttonSize))
			values.y = resetValue;
		ImGui::PopFont();
		ImGui::PopStyleColor(3);

		ImGui::SameLine();
		ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.2f");
		ImGui::PopItemWidth();
		ImGui::SameLine();

		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f });
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
		ImGui::PushFont(boldFont);
		if (ImGui::Button("Z", buttonSize))
			values.z = resetValue;
		ImGui::PopFont();
		ImGui::PopStyleColor(3);

		ImGui::SameLine();
		ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.2f");
		ImGui::PopItemWidth();

		ImGui::PopStyleVar();

		ImGui::Columns(1);

		ImGui::PopID();
	}

	template<typename Component, typename UIFunction>
	static void DrawComponent(const String& name, Entity* entity, UIFunction uiFunction)
	{
		const ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
		if (entity->HasComponent<Component>())
		{
			auto& component = entity->GetComponent<Component>();
			ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{ 4, 4 });
			float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
			ImGui::Separator();
			bool open = ImGui::TreeNodeEx((void*)typeid(Component).hash_code(), treeNodeFlags, name.c_str());
			ImGui::PopStyleVar();
			ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5f);
			if (ImGui::Button("+", ImVec2{ lineHeight, lineHeight }))
			{
				ImGui::OpenPopup("ComponentSettings");
			}

			bool removeComponent = false;
			if (ImGui::BeginPopup("ComponentSettings"))
			{
				if (ImGui::MenuItem("Remove component"))
					removeComponent = true;

				ImGui::EndPopup();
			}

			if (open)
			{
				uiFunction(component);
				ImGui::TreePop();
			}

			if (removeComponent)
				entity->RemoveComponent<Component>();
		}
	}

	void InspectorWindow::DrawComponents_Internal(Entity* entity)
	{
		auto& name = entity->mName;
		char buffer[256];
		memset(buffer, 0, sizeof(buffer));
		std::strncpy(buffer, name.c_str(), sizeof(buffer));
		if (ImGui::InputText("##Name", buffer, sizeof(buffer)))
		{
			name = String(buffer);
		}

		ImGui::Separator();
		static float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		static ImVec2 addComponentButtonSize = { ImGui::GetWindowWidth(), lineHeight };
		ImGui::SameLine(-1);
		if (ImGui::Button("Add Component", addComponentButtonSize))
		{
			ImGui::OpenPopup("AddComponent");
		}
		if (ImGui::BeginPopup("AddComponent"))
		{
			if (ImGui::MenuItem("Camera"))
			{
			}

			if (ImGui::MenuItem(""))
			{
			}
			ImGui::EndPopup();
		}

		DrawComponent<TransformComponent>("Transform", entity, [&entity](auto& component)
		{
			bool dirty = false;
			Vector3 translation; Quaternion rotation; Vector3 scale;
			MathUtil::DecomposeMatrix(component.localToParentMatrix, translation, rotation, scale);
			Vector3 eulerAngles = glm::degrees(glm::eulerAngles(rotation));
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
			ImGui::Columns(2);
			ImGui::Separator();

			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted("Position");
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			if (ImGui::DragFloat3("##Position", &translation.x, 0.1f))
			{
				dirty = true;
			}

			ImGui::PopItemWidth();
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted("Rotation");
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			if (ImGui::DragFloat3("##Rotation", &eulerAngles.x))
			{
				eulerAngles.y = std::min(eulerAngles.y, 89.9f);
				eulerAngles.y = std::max(eulerAngles.y, -89.9f);
				rotation = Quaternion(glm::radians(eulerAngles));
				dirty = true;
			}

			ImGui::PopItemWidth();
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted("Scale");
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			if (ImGui::DragFloat3("##Scale", &scale.x, 0.01f, 0.01f, std::numeric_limits<float>::max()))
			{
				scale = MathUtil::MaxVector3(Vector3(0.01f), scale);
				dirty = true;
			}

			ImGui::PopItemWidth();
			ImGui::NextColumn();

			ImGui::Columns(1);
			ImGui::Separator();
			ImGui::PopStyleVar();

			if (dirty)
			{
				entity->ReplaceComponent<TransformComponent>(MathUtil::ComposeMatrix(translation, rotation, scale));
			}
		});

		DrawComponent<StaticMeshComponent>("Static Mesh", entity, [&entity](StaticMeshComponent& component)
		{
			auto& mesh = component.mesh;
			ImGui::Columns(3);
			ImGui::SetColumnWidth(0, 100);
			ImGui::SetColumnWidth(1, 300);
			ImGui::SetColumnWidth(2, 40);
			ImGui::Text("Mesh");
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			if (component.mesh != nullptr)
			{
			
			}
			else
			{
				ImGui::InputText("##meshfilepath", (char*)"Null", 256, ImGuiInputTextFlags_ReadOnly);
			}
			ImGui::PopItemWidth();
			ImGui::NextColumn();
			if (ImGui::Button("...##openmesh"))
			{
			}
			ImGui::Columns(1);

			if (mesh != nullptr)
			{
				if (ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen))
				{
					auto& materials = mesh->GetMaterials();

					static uint32 selectedMaterialIndex = 0;
					for (uint32 i = 0; i < materials.size(); i++)
					{
						auto& material = materials[i];

						ImGuiTreeNodeFlags treeNodeFlags = (selectedMaterialIndex == i ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_Leaf;
						bool opened = ImGui::TreeNodeEx((void*)(&material), treeNodeFlags, "Material name");
						if (ImGui::IsItemClicked())
						{
							selectedMaterialIndex = i;
						}
						if (opened)
						{
							ImGui::TreePop();
						}
					}
				}
			}
		});

		DrawComponent<LightComponent>("Light", entity, [&entity](auto& component)
		{
			static const char* types[] = { "Directional", "Point", "Spot", "Area" };
		
			auto& light = component;
			int item_type = (int)light.type;
			// auto direction = light.direction;
			auto color = light.color;
			auto intensity = light.intensity;
			auto range = light.range;

			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
			ImGui::Columns(2);
			ImGui::Separator();

			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted("Type");
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			if (ImGui::Combo("##Type", &item_type, types, IM_ARRAYSIZE(types), IM_ARRAYSIZE(types)))
			{
				light.type = (LightType)item_type;
			}
			ImGui::PopItemWidth();
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted("Color");
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			if (ImGui::ColorEdit4("##Color", &color.x, ImGuiColorEditFlags_DisplayRGB))
			{
				light.color = color;
			}
			ImGui::PopItemWidth();
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted("Range");
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			if (ImGui::DragFloat("##Range", &range, 0.1f))
			{
				light.range = range;
			}
			ImGui::PopItemWidth();
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted("Intensity");
			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			if (ImGui::DragFloat("##Intensity", &intensity, 0.1f))
			{
				light.intensity = intensity;
			}
			ImGui::PopItemWidth();
			ImGui::NextColumn();

			ImGui::Columns(1);
			ImGui::Separator();
			ImGui::PopStyleVar();
		});
		
		DrawComponent<CameraComponent>("Camera", entity, [&entity](auto& component)
		{
			auto camera = component.camera.get();

			float yFov = glm::degrees(camera->GetVerticalFieldOfView());
			if (ImGui::DragFloat("Vertical Field Of View", &yFov))
			{
				camera->SetVerticalFieldOfView(glm::radians(yFov));
			}

			float nearZ = camera->GetNearPlane();
			if (ImGui::DragFloat("Near Plane", &nearZ))
			{
				camera->SetNearPlane(nearZ);
			}

			float farZ = camera->GetFarPlane();
			if (ImGui::DragFloat("Far Plane", &farZ))
			{
				camera->SetFarPlane(farZ);
			}
		});
	}

	void InspectorWindow::Draw()
	{
		if (ImGui::Begin(mTitle.c_str(), &mActive))
		{
			if (mEditor->mSceneContext.selectedEntity != nullptr)
			{
				DrawComponents_Internal(mEditor->mSceneContext.selectedEntity);
			}
		}
		ImGui::End();
	}
}