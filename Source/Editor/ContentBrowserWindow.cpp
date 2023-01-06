#include "ContentBrowserWindow.h"

#include <imgui/imgui_internal.h>
#include <shellapi.h>

namespace HE
{
	std::string ToLower(const std::string& string)
	{
		std::string result;
		for (const auto& character : string)
		{
			result += std::tolower(character);
		}
		return result;
	}

	struct FileSystem
	{
		static bool Exists(const std::filesystem::path& filepath);
		static bool ShowFileInExplorer(const std::filesystem::path& path);
		static bool OpenExternally(const std::filesystem::path& path);
		static bool DeleteFile(const std::filesystem::path& filepath);
		static bool CreateDirectory(const std::filesystem::path& directory);
		static bool OpenDirectoryInExplorer(const std::filesystem::path& path);
		static bool Rename(const std::filesystem::path& oldPath, const std::filesystem::path& newPath);
	};

	bool FileSystem::Exists(const std::filesystem::path& filepath)
	{
		return std::filesystem::exists(filepath);
	}

	bool FileSystem::ShowFileInExplorer(const std::filesystem::path& path)
	{
		auto absolutePath = std::filesystem::canonical(path);
		if (!Exists(absolutePath))
		{
			return false;
		}
		std::string cmd = fmt::format("explorer.exe /select,\"{0}\"", absolutePath.string());
		system(cmd.c_str());
		return true;
	}

	bool FileSystem::Rename(const std::filesystem::path& oldPath, const std::filesystem::path& newPath)
	{
		std::filesystem::rename(oldPath, newPath);
		return true;
	}

	bool FileSystem::OpenExternally(const std::filesystem::path& path)
	{
		auto absolutePath = std::filesystem::canonical(path);
		if (!Exists(absolutePath))
		{
			return false;
		}
		ShellExecute(NULL, L"open", absolutePath.c_str(), NULL, NULL, SW_SHOWNORMAL);
		return true;
	}

	bool FileSystem::DeleteFile(const std::filesystem::path& filepath)
	{
		if (!FileSystem::Exists(filepath))
		{
			return false;
		}
		if (std::filesystem::is_directory(filepath))
		{
			return std::filesystem::remove_all(filepath) > 0;
		}
		return std::filesystem::remove(filepath);
	}

	bool FileSystem::CreateDirectory(const std::filesystem::path& directory)
	{
		return std::filesystem::create_directories(directory);
	}

	bool FileSystem::OpenDirectoryInExplorer(const std::filesystem::path& path)
	{
		auto absolutePath = std::filesystem::canonical(path);
		if (!Exists(absolutePath))
		{
			return false;
		}
		ShellExecute(NULL, L"explore", absolutePath.c_str(), NULL, NULL, SW_SHOWNORMAL);
		return true;
	}
}

namespace Horizon::UI
{
	static int s_UIContextID = 0;
	static uint32_t s_Counter = 0;
	static char s_IDBuffer[16];

	static void PushID()
	{
		ImGui::PushID(s_UIContextID++);
		s_Counter = 0;
	}

	static void PopID()
	{
		ImGui::PopID();
		s_UIContextID--;
	}

	static void BeginPropertyGrid()
	{
		PushID();
		ImGui::Columns(2);
	}

	static void Separator()
	{
		ImGui::Separator();
	}

	static void PushItemDisabled()
	{
		ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
	}

	static void PopItemDisabled()
	{
		ImGui::PopItemFlag();
	}

	static void EndPropertyGrid()
	{
		ImGui::Columns(1);
		PopID();
	}

	static bool TreeNode(const std::string& id, const std::string& label, ImGuiTreeNodeFlags flags = 0)
	{
		ImGuiWindow* window = ImGui::GetCurrentWindow();
		if (window->SkipItems)
		{
			return false;
		}
		return ImGui::TreeNodeBehavior(window->GetID(id.c_str()), flags, label.c_str(), NULL);
	}

	std::map<SharedPtr<Texture>, ImTextureID> texMap;
	static bool ImageButton(const char* stringID, SharedPtr<Texture> texture, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), int frame_padding = -1, const ImVec4& bg_col = ImVec4(0, 0, 0, 0), const ImVec4& tint_col = ImVec4(1, 1, 1, 1))
	{
		ImTextureID textureID;
		if (texMap.find(texture) == texMap.end())
		{
			const auto& view = texture->GetImage()->GetView(ImageViewInfo(0, 1, 0, 1))->GetHandle();
			textureID = ImGui_ImplVulkan_AddTexture(texture->GetSampler()->GetHandle(), view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			texMap[texture] = textureID;
		}
		else
		{
			textureID = texMap[texture];
		}
		return ImGui::ImageButtonEx(ImGui::GetID(stringID), textureID, size, uv0, uv1, ImVec2{ (float)frame_padding, (float)frame_padding }, bg_col, tint_col);
	}
}

namespace HE
{

void ContentBrowserWindow::SelectItem(ContentBrowserItem* item)
{
	item->isSelected = true;
	selectedItems.Add(item->GetUuid());
}

void ContentBrowserWindow::DeselectItem(ContentBrowserItem* item)
{
	item->isSelected = false;
	selectedItems.Remove(item->GetUuid());
	item->isRenaming = false;
	memset(renameBuffer, 0, MAX_INPUT_BUFFER_LENGTH);
}

void ContentBrowserWindow::ClearSelections()
{
	for (auto& item : currentItems)
	{
		item.isSelected = false;
	}
	selectedItems.Clear();
}

void ContentBrowserWindow::ChangeDirectory(ContentBrowserDirectory* directory)
{
	if (!directory)
	{
		return;
	}

	m_UpdateNavigationPath = true;

	currentItems.Clear();
	{
		for (const auto& [uuid, subdir] : directory->subdirectories)
		{
			currentItems.Add(std::move(ContentBrowserDirectory(uuid, subdir->path, directory, directoryIcon)));
		}
		std::vector<UUID> invalidAssets;
		for (const auto& uuid : directory->assets)
		{
			const auto& assetFile = assetManager->GetAssetFile(uuid);
			if (!assetFile)
			{
				// invalidAssets.emplace_back(assetFile->uuid);
			}
			else
			{
				const auto& icon = AssetIconMap.find(assetFile->path.extension().string()) != AssetIconMap.end() ? AssetIconMap[assetFile->path.extension().string()] : fileIcon;
				currentItems.Add(std::move(ContentBrowserAsset(uuid, assetFile->path, assetFile->type, icon)));
			}
		}
		for (auto invalidHandle : invalidAssets)
		{
			directory->assets.erase(std::remove(directory->assets.begin(), directory->assets.end(), invalidHandle), directory->assets.end());
		}
	}

	SortItemList();
	ClearSelections();

	previousDirectory = currentDirectory;
	currentDirectory = directory;
}

void ContentBrowserWindow::UpdateDropArea(ContentBrowserDirectory* directory)
{
	if ((directory->uuid != currentDirectory->uuid) && ImGui::BeginDragDropTarget())
	{
		const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("asset_payload");
		if (payload)
		{
			uint32 count = payload->DataSize / sizeof(UUID);
			for (uint32 i = 0; i < count; i++)
			{
				UUID uuid = *(((UUID*)payload->Data) + i);
				uint32 index = currentItems.Find(uuid);
				if (index != ContentBrowserItemList::InvalidIndex)
				{
					// currentItems[index].Move(directory->path);
					currentItems.Remove(uuid);
				}
			}
		}
		ImGui::EndDragDropTarget();
	}
}

void ContentBrowserWindow::RenderDirectoryHierarchy(ContentBrowserDirectory* directory)
{
	std::string name = directory->path.filename().string();
	std::string id = name + "_TreeNode";
	bool previousState = ImGui::TreeNodeBehaviorIsOpen(ImGui::GetID(id.c_str()));
	bool open = UI::TreeNode(id, name, ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnDoubleClick);
	if (open)
	{
		for (auto& [handle, child] : directory->subdirectories)
		{
			RenderDirectoryHierarchy(child);
		}
	}
	UpdateDropArea(directory);
	if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && directory->uuid != currentDirectory->uuid)
	{
		if (!ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.01f))
		{
			ChangeDirectory(directory);
		}
	}
	if (open)
	{
		ImGui::TreePop();
	}
}

ContentBrowserItemList ContentBrowserWindow::Search(const std::string& content, ContentBrowserDirectory* directory)
{
	ContentBrowserItemList results;
	std::string contentLowerCase = ToLower(content);

	for (auto& [uuid, subdir] : directory->subdirectories)
	{
		std::string subdirName = ToLower(subdir->GetName());
		if (subdirName.find(contentLowerCase) != std::string::npos)
		{
			results.Add(std::move(ContentBrowserDirectory(uuid, subdir->path, directory, directoryIcon)));
		}
		ContentBrowserItemList list = Search(content, subdir);
		results.items.insert(results.items.end(), list.items.begin(), list.items.end());
	}

	for (auto& uuid : directory->assets)
	{
		const auto& assetFile = assetManager->GetAssetFile(uuid);
		if (assetFile)
		{
			if (contentLowerCase[0] == '.')
			{
				if (assetFile->path.extension().string().find(std::string(&contentLowerCase[1])) != std::string::npos)
				{
					const auto& icon = AssetIconMap.find(assetFile->path.extension().string()) != AssetIconMap.end() ? AssetIconMap[assetFile->path.extension().string()] : fileIcon;
					results.Add(std::move(ContentBrowserAsset(uuid, assetFile->path, assetFile->type, icon)));
				}
			}
			else
			{
				std::string stem = ToLower(assetFile->path.stem().string());
				if (stem.find(contentLowerCase) != std::string::npos)
				{
					const auto& icon = AssetIconMap.find(assetFile->path.extension().string()) != AssetIconMap.end() ? AssetIconMap[assetFile->path.extension().string()] : fileIcon;
					results.Add(std::move(ContentBrowserAsset(uuid, assetFile->path, assetFile->type, icon)));
				}
			}
		}
		else
		{
			HE_LOG_ERROR("Failed to find asset from Asset Manager.");
		}
	}

	return results;
}

void ContentBrowserWindow::RenderTopBar()
{
	ImGui::BeginChild("##top_bar", ImVec2(0, 30));
	{
		if (UI::ImageButton("##back_button", backButtonIcon, ImVec2(25, 25)) && previousDirectory->uuid != baseDirectory->uuid)
		{
			nextDirectory = currentDirectory;
			previousDirectory = currentDirectory->parent;
			ChangeDirectory(previousDirectory);
		}
		ImGui::SameLine();
		if (UI::ImageButton("##forward_button", forwardButtonIcon, ImVec2(25, 25)))
		{
			ChangeDirectory(nextDirectory);
		}
		ImGui::SameLine();
		if (UI::ImageButton("##refresh_button", refreshButtonIcon, ImVec2(25, 25)))
		{
			Refresh();
		}
		ImGui::SameLine();
		if (m_UpdateNavigationPath)
		{
			m_BreadCrumbData.clear();
			ContentBrowserDirectory* current = currentDirectory;
			while (current && current->parent != nullptr)
			{
				m_BreadCrumbData.push_back(current);
				current = current->parent;
			}
			std::reverse(m_BreadCrumbData.begin(), m_BreadCrumbData.end());
			m_UpdateNavigationPath = false;
		}
		const std::string& assetsDirectoryName = baseDirectory->path.string();
		ImVec2 textSize = ImGui::CalcTextSize(assetsDirectoryName.c_str());
		if (ImGui::Selectable(assetsDirectoryName.c_str(), false, 0, ImVec2(textSize.x, 22)))
		{
			ChangeDirectory(baseDirectory);
		}
		ImGui::SameLine(ImGui::GetContentRegionAvail().x - searchBarWidth);
		{
			if (ImGui::InputTextWithHint("", "Search (Ctrl+F)", searchBuffer, MAX_INPUT_BUFFER_LENGTH))
			{
				if (strlen(searchBuffer) == 0)
				{
					ChangeDirectory(currentDirectory);
				}
				else
				{
					currentItems = Search(searchBuffer, currentDirectory);
					SortItemList();
				}
			}
		}

		UpdateDropArea(baseDirectory);

		ImGui::SameLine();
		for (auto& directory : m_BreadCrumbData)
		{
			ImGui::Text("/");
			ImGui::SameLine();
			std::string directoryName = directory->path.filename().string();
			ImVec2 textSize = ImGui::CalcTextSize(directoryName.c_str());
			if (ImGui::Selectable(directoryName.c_str(), false, 0, ImVec2(textSize.x, 22)))
			{
				ChangeDirectory(directory);
			}
			UpdateDropArea(directory);
			ImGui::SameLine();
		}
	}
	ImGui::EndChild();
}

void ContentBrowserWindow::RenderItems()
{
	isAnyItemHovered = false;
	std::lock_guard<std::mutex> lock(lockMutex);
	for (auto& item : currentItems)
	{
		ContentBrowserActionFlags result = ContentBrowserActionFlags::None;

		ImGui::PushID(ImGui::GetID(item.GetName().c_str()));
		ImGui::BeginGroup();

		if (item.IsSelected())
		{
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.25f, 0.25f, 0.75f));
		}
		UI::ImageButton(item.GetName().c_str(), item.GetIcon(), { thumbnailSize, thumbnailSize });
		if (item.IsSelected())
		{
			ImGui::PopStyleColor();
		}
		if (item.GetType() == ContentBrowserItemType::Directory && !item.IsSelected())
		{
			if (ImGui::BeginDragDropTarget())
			{
				const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("asset_payload");
				if (payload)
				{
					uint32 count = payload->DataSize / sizeof(UUID);
					for (uint32 i = 0; i < count; i++)
					{
						UUID uuid = *(((UUID*)payload->Data) + i);
						uint32 index = currentItems.Find(uuid);
						if (index != ContentBrowserItemList::InvalidIndex)
						{
							/*if (currentItems[index]->Move(m_DirectoryInfo->FilePath))
							{
								actionResult.Set(ContentBrowserAction::Refresh, true);
								currentItems.erase(assetHandle);
							}*/
						}
					}
				}
				ImGui::EndDragDropTarget();
			}
		}

		bool dragging = false;
		if (dragging = ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
		{
			item.isDragging = true;
			if (!selectedItems.Contains(item.GetUuid()))
			{
				result |= ContentBrowserActionFlags::ClearSelections;
			}
			if (selectedItems.GetCount() > 0)
			{
				for (const auto& uuid : selectedItems)
				{
					uint32 index = currentItems.Find(uuid);
					if (index == ContentBrowserItemList::InvalidIndex)
					{
						continue;
					}
					const auto& item = currentItems[index];
					// UI::Image(item.GetIcon(), ImVec2(20, 20));
					ImGui::SameLine();
					const auto& name = item.GetName();
					ImGui::TextUnformatted(name.c_str());
				}
				ImGui::SetDragDropPayload("asset_payload", selectedItems.GetData(), sizeof(UUID) * selectedItems.GetCount());
			}
			result |= ContentBrowserActionFlags::Selected;
			ImGui::EndDragDropSource();
		}

		if (ImGui::IsItemHovered())
		{
			result |= ContentBrowserActionFlags::Hovered;
			if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
			{
				// Activate(result);
			}
			else
			{
				bool action = selectedItems.GetCount() > 1 ? ImGui::IsMouseReleased(ImGuiMouseButton_Left) : ImGui::IsMouseClicked(ImGuiMouseButton_Left);
				bool skipBecauseDragging = item.isDragging && selectedItems.Contains(item.GetUuid());
				if (action && !skipBecauseDragging)
				{
					result |= ContentBrowserActionFlags::Selected;
					if (!Input::IsKeyPressed(KeyCode::LeftControl) && !Input::IsKeyPressed(KeyCode::LeftShift))
					{
						result |= ContentBrowserActionFlags::ClearSelections;
					}
					if (Input::IsKeyPressed(KeyCode::LeftShift))
					{
						result |= ContentBrowserActionFlags::SelectToHere;
					}
				}
			}
		}
		if (ImGui::BeginPopupContextItem("ContentBrowserItemContextMenu"))
		{
			result |= ContentBrowserActionFlags::Selected;
			if (ImGui::MenuItem("Open", "Ctrl+Shift+O"))
			{

			}
			ImGui::Separator();
			if (ImGui::MenuItem("Cut", "Ctrl+X"))
			{

			}
			if (ImGui::MenuItem("Copy", "Ctrl+C"))
			{

			}
			if (ImGui::MenuItem("Paste", "Ctrl+V"))
			{

			}
			if (ImGui::MenuItem("Duplicate", "Ctrl+D"))
			{

			}
			ImGui::Separator();
			if (ImGui::MenuItem("Save", "Ctrl+S"))
			{

			}
			if (ImGui::MenuItem("Rename", "F2"))
			{
				StartRenamingItem(&item);
			}
			if (ImGui::MenuItem("Delete", "Delete"))
			{
				result |= ContentBrowserActionFlags::DeleteSelectedItems;
			}
			ImGui::Separator();
			if (ImGui::MenuItem("Show In Explorer"))
			{
				result |= ContentBrowserActionFlags::ShowInExplorer;
			}
			if (ImGui::MenuItem("Open Externally"))
			{
				result |= ContentBrowserActionFlags::OpenExternally;
			}
			// item.RenderCustomContextItems();
			ImGui::EndPopup();
		}
		if (!item.isRenaming)
		{
			ImGui::TextWrapped(item.GetName().c_str());
			if (Input::IsKeyPressed(KeyCode::F2) && item.IsSelected())
			{
				StartRenamingItem(&item);
			}
		}
		else
		{
			ImGui::SetKeyboardFocusHere();
			if (!item.IsSelected() || ImGui::InputText("##rename", renameBuffer, MAX_INPUT_BUFFER_LENGTH, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll))
			{
				const auto& newPath = item.path.parent_path() / renameBuffer;
				if (FileSystem::Exists(newPath))
				{
					HE_LOG_ERROR("Already exists!");
					item.isRenaming = false;
				}
				else
				{
					FileSystem::Rename(item.path, newPath);
					item.name = renameBuffer;
					item.isRenaming = false;
					ProcessDirectory(baseDirectory->path, nullptr);
					ChangeDirectory(currentDirectory);
					result |= ContentBrowserActionFlags::Renamed;
				}
			}
		}
		item.isDragging = dragging;

		ImGui::EndGroup();
		ImGui::PopID();
		ImGui::NextColumn();

		if (HAS_ANY_FLAGS(result, ContentBrowserActionFlags::ClearSelections))
		{
			ClearSelections();
		}
		if (HAS_ANY_FLAGS(result, ContentBrowserActionFlags::Selected) && !selectedItems.Contains(item.GetUuid()))
		{
			SelectItem(&item);
		}
		if (HAS_ANY_FLAGS(result, ContentBrowserActionFlags::Deselected) && selectedItems.Contains(item.GetUuid()))
		{
			DeselectItem(&item);
		}
		if (HAS_ANY_FLAGS(result, ContentBrowserActionFlags::SelectToHere) && selectedItems.GetCount() == 2)
		{
			uint32 firstIndex = currentItems.Find(selectedItems[0]);
			uint32 lastIndex = currentItems.Find(item.GetUuid());
			if (firstIndex > lastIndex)
			{
				uint32 temp = firstIndex;
				firstIndex = lastIndex;
				lastIndex = temp;
			}
			for (uint32 i = firstIndex + 1; i < lastIndex; i++)
			{
				SelectItem(&currentItems[i]);
			}
		}
		if (HAS_ANY_FLAGS(result, ContentBrowserActionFlags::ShowInExplorer))
		{
			if (item.GetType() == ContentBrowserItemType::Directory)
			{
				FileSystem::OpenDirectoryInExplorer(item.GetPath());
			}
			else
			{
				FileSystem::ShowFileInExplorer(item.GetPath());
			}
		}
		if (HAS_ANY_FLAGS(result, ContentBrowserActionFlags::OpenExternally))
		{
			FileSystem::OpenExternally(item.GetPath());
		}
		if (HAS_ANY_FLAGS(result, ContentBrowserActionFlags::Hovered))
		{
			isAnyItemHovered = true;
		}

		if (HAS_ANY_FLAGS(result, ContentBrowserActionFlags::DeleteSelectedItems))
		{
			DeleteSelectedItems();
			break;
		}
		if (HAS_ANY_FLAGS(result, ContentBrowserActionFlags::Renamed))
		{
			SortItemList();
			break;
		}
		if (HAS_ANY_FLAGS(result, ContentBrowserActionFlags::ChangeDirectory))
		{
			ChangeDirectory((ContentBrowserDirectory*)&item);
			break;
		}
		if (HAS_ANY_FLAGS(result, ContentBrowserActionFlags::Refresh))
		{
			Refresh();
			break;
		}
	}
}

void ContentBrowserWindow::RenderBottomBar()
{
	ImGui::BeginChild("##panel_controls", ImVec2(ImGui::GetColumnWidth() - 12, 30), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
	{
		ImGui::Separator();
		ImGui::Columns(4, 0, false);
		if (selectedItems.GetCount() == 1)
		{
			std::string path;
			const auto& assetFile = assetManager->GetAssetFile(selectedItems[0]);
			if (assetFile)
			{
				path = assetFile->path.string();
			}
			else if (directories.find(selectedItems[0]) != directories.end())
			{
				path = directories[selectedItems[0]]->path.string();
				std::replace(path.begin(), path.end(), '\\', '/');
			}
			ImGui::Text(path.c_str());
		}
		else if (selectedItems.GetCount() > 1)
		{
			ImGui::Text("%d items selected", selectedItems.GetCount());
		}
		ImGui::NextColumn();
		ImGui::NextColumn();
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(ImGui::GetColumnWidth());
		ImGui::SliderFloat("##thumbnail_size", &thumbnailSize, 32.0f, 96.0f);
	}
	ImGui::EndChild();
}

void ContentBrowserWindow::OnImGuiRender()
{
	ImGui::Begin("Content Browser", NULL, ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar);
	isHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);
	isFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
	{
		UI::BeginPropertyGrid();
		ImGui::SetColumnOffset(1, 300.0f);
		ImGui::BeginChild("##folders_common");
		{
			bool open = ImGui::CollapsingHeader("Content", nullptr, ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnDoubleClick);
			if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && baseDirectory->uuid != currentDirectory->uuid)
			{
				if (!ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.01f))
				{
					ChangeDirectory(baseDirectory);
				}
			}
			if (open)
			{
				for (auto& [handle, directory] : baseDirectory->subdirectories)
				{
					RenderDirectoryHierarchy(directory);
				}
			}

		}
		ImGui::EndChild();
		ImGui::NextColumn();
		ImGui::BeginChild("##directory_structure", ImVec2(0, ImGui::GetWindowHeight() - 65));
		{
			RenderTopBar();
			ImGui::Separator();
			ImGui::BeginChild("Scrolling");
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.35f));
				if (ImGui::BeginPopupContextWindow(0, 1, false))
				{
					if (ImGui::BeginMenu("New"))
					{
						if (ImGui::MenuItem("Folder"))
						{
							const auto& path = currentDirectory->path / "New Folder";
							bool created = FileSystem::CreateDirectory(path);
							if (created)
							{
								const auto& directory = new ContentBrowserDirectory(UUID::Generate(), path, currentDirectory, directoryIcon);
								directories[directory->uuid] = directory;
								auto& newFolder = currentItems.Add(*directory);
								StartRenamingItem(&newFolder);
								ClearSelections();
								SelectItem(&newFolder);
								SortItemList();
							}
						}
						if (ImGui::MenuItem("Scene"))
						{
							//
						}
						if (ImGui::MenuItem("Material"))
						{
							//
						}
						ImGui::EndMenu();
					}
					if (ImGui::MenuItem("Refresh"))
					{
						Refresh();
					}
					ImGui::Separator();
					if (ImGui::MenuItem("Show in Explorer"))
					{
						FileSystem::OpenDirectoryInExplorer(currentDirectory->path);
					}
					ImGui::EndPopup();
				}
				float cellSize = thumbnailSize + padding;
				float panelWidth = ImGui::GetContentRegionAvail().x;
				int columnCount = (int)(panelWidth / cellSize);
				if (columnCount < 1) columnCount = 1;
				ImGui::Columns(columnCount, 0, false);
				RenderItems();
				if (ImGui::IsWindowFocused() && !ImGui::IsMouseDragging(ImGuiMouseButton_Left))
				{
					UpdateInput();
				}
				ImGui::PopStyleColor(2);
			}
			ImGui::EndChild();
		}
		ImGui::EndChild();
		if (ImGui::BeginDragDropTarget())
		{
			auto data = ImGui::AcceptDragDropPayload("scene_hierarchy");
			if (data)
			{
				//
			}
			ImGui::EndDragDropTarget();
		}
		RenderBottomBar();
		UI::EndPropertyGrid();
	}
	ImGui::End();
}

void ContentBrowserWindow::DeleteSelectedItems()
{
	for (UUID uuid : selectedItems)
	{
		uint32 index = currentItems.Find(uuid);
		if (index == ContentBrowserItemList::InvalidIndex)
		{
			continue;
		}
		auto& item = currentItems[index];
		bool deleted = FileSystem::DeleteFile(item.path);
		if (!deleted)
		{
			HE_LOG_ERROR("Failed to delete {0}", item.path.string());
			return;
		}
		switch (item.GetType())
		{
		case ContentBrowserItemType::Directory:
			break;
		case ContentBrowserItemType::Asset:
			break;
		}
		currentItems.Remove(uuid);
	}
	ProcessDirectory(baseDirectory->path, nullptr);
	ChangeDirectory(currentDirectory);
}

void ContentBrowserWindow::SortItemList()
{
	std::sort(currentItems.begin(), currentItems.end(),
		[](const ContentBrowserItem& item1, const ContentBrowserItem& item2)
		{
			if (item1.GetType() == item2.GetType())
			{
				return ToLower(item1.GetName()) < ToLower(item2.GetName());
			}
			return (uint32)item1.GetType() < (uint32)item2.GetType();
		});
}

ContentBrowserDirectory* ContentBrowserWindow::GetDirectory(const std::filesystem::path& path) const
{
	for (const auto& [uuid, directory] : directories)
	{
		if (directory->path == path)
		{
			return directory;
		}
	}
	return nullptr;
}

UUID ContentBrowserWindow::ProcessDirectory(const std::filesystem::path& path, ContentBrowserDirectory* parent)
{
	ContentBrowserDirectory* directory = GetDirectory(path);
	if (directory)
	{
		directory->assets.clear();
		directory->subdirectories.clear();
	}
	else
	{
		directory = new ContentBrowserDirectory(UUID::Generate(), path, parent, directoryIcon);
	}
	for (auto entry : std::filesystem::directory_iterator(path))
	{
		if (entry.is_directory())
		{
			UUID uuid = ProcessDirectory(entry.path(), directory);
			directory->subdirectories[uuid] = directories[uuid];
		}
		else
		{
			const auto& assetFile = assetManager->GetAssetFile(entry.path());
			if (!assetFile)
			{
				UUID uuid = assetManager->ImportAsset(entry.path());
				if (uuid)
				{
					directory->assets.push_back(uuid);
				}
			}
			else
			{
				directory->assets.push_back(assetFile->uuid);
			}
		}
	}
	directories[directory->uuid] = directory;
	return directory->uuid;
}

void ContentBrowserWindow::Refresh()
{
	ProcessDirectory(baseDirectory->path, nullptr);
	ChangeDirectory(currentDirectory);
}

void ContentBrowserWindow::UpdateInput()
{
	if (!isHovered)
	{
		return;
	}

	if (!isAnyItemHovered && ImGui::IsAnyMouseDown())
	{
		ClearSelections();
	}

	if (Input::IsKeyPressed(KeyCode::Delete))
	{
		DeleteSelectedItems();
	}

	if (Input::IsKeyPressed(KeyCode::F5))
	{
		Refresh();
	}
}

void ContentBrowserWindow::RenameSelectedItems()
{

}

void ContentBrowserWindow::StartRenamingItem(ContentBrowserItem* item)
{
	if (item->isRenaming)
	{
		return;
	}
	memset(renameBuffer, 0, MAX_INPUT_BUFFER_LENGTH);
	memcpy(renameBuffer, item->name.c_str(), item->name.size());
	item->isRenaming = true;
}

ContentBrowserWindow::ContentBrowserWindow(AssetManager* assetManager)
	: assetManager(assetManager)
{
	fileIcon = Texture::Create2DFromFile("D:/Programming/CBTEST/Icons/file.png");
	directoryIcon = Texture::Create2DFromFile("D:/Programming/CBTEST/Icons/folder.png");
	backButtonIcon = Texture::Create2DFromFile("D:/Programming/CBTEST/Icons/btn_back.png");
	forwardButtonIcon = Texture::Create2DFromFile("D:/Programming/CBTEST/Icons/btn_fwrd.png");
	refreshButtonIcon = Texture::Create2DFromFile("D:/Programming/CBTEST/Icons/refresh.png");


	AssetIconMap[".fbx"] = Texture::Create2DFromFile("D:/Programming/CBTEST/Icons/fbx.png");
	AssetIconMap[".obj"] = Texture::Create2DFromFile("D:/Programming/CBTEST/Icons/obj.png");
	AssetIconMap[".wav"] = Texture::Create2DFromFile("D:/Programming/CBTEST/Icons/wav.png");
	AssetIconMap[".cs"] = Texture::Create2DFromFile("D:/Programming/CBTEST/Icons/cs.png");
	AssetIconMap[".png"] = Texture::Create2DFromFile("D:/Programming/CBTEST/Icons/png.png");
	AssetIconMap[".hmaterial"] = Texture::Create2DFromFile("D:/Programming/CBTEST/Icons/MaterialAssetIcon.png");
	AssetIconMap[".hscene"] = Texture::Create2DFromFile("D:/Programming/CBTEST/Icons/hazel.png");

	UUID uuid = ProcessDirectory("D:/Programming/CBTEST", nullptr);
	baseDirectory = directories[uuid];
	ChangeDirectory(baseDirectory);

	memset(searchBuffer, 0, MAX_INPUT_BUFFER_LENGTH);
}

ContentBrowserWindow::~ContentBrowserWindow()
{
}

}
