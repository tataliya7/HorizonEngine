#pragma once

#include <HorizonEngine.h>

#include <mutex>
#include <filesystem>

namespace HE
{

enum class ContentBrowserActionFlags
{
	None = 0,
	Refresh = 1 << 0,
	ClearSelections = 1 << 1,
	Selected = 1 << 2,
	Deselected = 1 << 3,
	Hovered = 1 << 4,
	Renamed = 1 << 5,
	ChangeDirectory = 1 << 6,
	DeleteSelectedItems = 1 << 7,
	SelectToHere = 1 << 8,
	Moved = 1 << 9,
	ShowInExplorer = 1 << 10,
	OpenExternally = 1 << 11,
	Reload = 1 << 12,
};
ENUM_CLASS_OPERATORS(ContentBrowserActionFlags);

enum class ContentBrowserItemType
{
	Directory,
	Asset,
};

#define MAX_INPUT_BUFFER_LENGTH 128
class ContentBrowserItem
{
public:
	ContentBrowserItem(UUID uuid, const std::string& name, const std::filesystem::path& path, ContentBrowserItemType type, SharedPtr<Texture> icon)
		: uuid(uuid), name(name), path(path), type(type), icon(icon) {}
	virtual ~ContentBrowserItem() {}
	bool IsSelected() const { return isSelected; }
	UUID GetUuid() const { return uuid; }
	ContentBrowserItemType GetType() const { return type; }
	const std::string& GetName() const { return name; }
	SharedPtr<Texture> GetIcon() const { return icon; }
	const std::filesystem::path& GetPath() const { return path; }
private:
	friend class ContentBrowserWindow;
	virtual void RenderCustomContextItems() {}
protected:
	UUID uuid;
	std::string name;
	std::filesystem::path path;
	ContentBrowserItemType type;
	SharedPtr<Texture> icon;
	bool isSelected = false;
	bool isRenaming = false;
	bool isDragging = false;
};

class ContentBrowserDirectory : public ContentBrowserItem
{
public:
	ContentBrowserDirectory(UUID uuid, std::filesystem::path path, ContentBrowserDirectory* parent, SharedPtr<Texture> icon)
		: ContentBrowserItem(uuid, path.filename().string(), path, ContentBrowserItemType::Directory, icon), parent(parent) {}
	~ContentBrowserDirectory() {}
private:
	friend class ContentBrowserWindow;
	ContentBrowserDirectory* parent;
	std::vector<UUID> assets;
	std::unordered_map<UUID, ContentBrowserDirectory*> subdirectories;
};

class ContentBrowserAsset : public ContentBrowserItem
{
public:
	ContentBrowserAsset(UUID uuid, std::filesystem::path path, AssetType type, SharedPtr<Texture> icon)
		: ContentBrowserItem(uuid, path.stem().string(), path, ContentBrowserItemType::Asset, icon), type(type) {}
	~ContentBrowserAsset() {}
private:
	friend class ContentBrowserWindow;
	AssetType type;
};

class ContentBrowserSelectionStack
{
public:
	void CopyFrom(const ContentBrowserSelectionStack& other)
	{
		items.assign(other.begin(), other.end());
	}
	bool Contains(UUID uuid) const
	{
		for (const auto& id : items)
		{
			if (id == uuid)
			{
				return true;
			}
		}
		return false;
	}
	void Add(UUID uuid)
	{
		if (Contains(uuid))
		{
			return;
		}
		items.push_back(uuid);
	}
	void Remove(UUID uuid)
	{
		if (!Contains(uuid))
		{
			return;
		}
		for (auto it = items.begin(); it != items.end(); it++)
		{
			if (uuid == *it)
			{
				items.erase(it);
				break;
			}
		}
	}
	void Clear()
	{
		items.clear();
	}
	uint32 GetCount() const { return (uint32)items.size(); }
	const UUID* GetData() const { return items.data(); }
	UUID operator[](uint32 index) const { return items[index]; }
	std::vector<UUID>::iterator begin() { return items.begin(); }
	std::vector<UUID>::iterator end() { return items.end(); }
	std::vector<UUID>::const_iterator begin() const { return items.begin(); }
	std::vector<UUID>::const_iterator end() const { return items.end(); }
private:
	std::vector<UUID> items;
};

struct ContentBrowserItemList
{
	static const uint32 InvalidIndex = std::numeric_limits<uint32>::max();
	std::vector<ContentBrowserItem> items;
	std::vector<ContentBrowserItem>::iterator begin() { return items.begin(); }
	std::vector<ContentBrowserItem>::iterator end() { return items.end(); }
	std::vector<ContentBrowserItem>::const_iterator begin() const { return items.begin(); }
	std::vector<ContentBrowserItem>::const_iterator end() const { return items.end(); }
	ContentBrowserItem& operator[](uint32 index) { return items[index]; }
	const ContentBrowserItem& operator[](uint32 index) const { return items[index]; }
	void Clear()
	{
		items.clear();
	}
	ContentBrowserItem& Add(ContentBrowserDirectory item)
	{
		return items.emplace_back(item);
	}
	ContentBrowserItem& Add(ContentBrowserAsset item)
	{
		return items.emplace_back(item);
	}
	void Remove(UUID uuid)
	{
		uint32 index = Find(uuid);
		if (index == InvalidIndex)
		{
			return;
		}
		items.erase(items.begin() + index);
	}
	uint32 Find(UUID uuid) const
	{
		for (uint32 i = 0; i < (uint32)items.size(); i++)
		{
			if (items[i].GetUuid() == uuid)
			{
				return i;
			}
		}
		return InvalidIndex;
	}
};

class ContentBrowserWindow
{
public:
	ContentBrowserWindow(AssetManager* assetManager);
	~ContentBrowserWindow();
	void UpdateDropArea(ContentBrowserDirectory* directory);
	void ChangeDirectory(ContentBrowserDirectory* directory);
	void RenderDirectoryHierarchy(ContentBrowserDirectory* directory);
	void RenderTopBar();
	void RenderItems();
	void RenderBottomBar();
	void OnImGuiRender();
	void Refresh();
	void UpdateInput();
	void RenameSelectedItems();
	void DeleteSelectedItems();
	void StartRenamingItem(ContentBrowserItem* item);
	void SelectItem(ContentBrowserItem* item);
	void DeselectItem(ContentBrowserItem* item);
	void ClearSelections();
	void SortItemList();
	ContentBrowserItemList Search(const std::string& content, ContentBrowserDirectory* directory);
private:
	ContentBrowserDirectory* GetDirectory(const std::filesystem::path& path) const;
	UUID ProcessDirectory(const std::filesystem::path& path, ContentBrowserDirectory* parent);
	AssetManager* assetManager;
	std::mutex lockMutex;
	bool isHovered;
	bool isFocused;
	bool isAnyItemHovered;
	ContentBrowserItemList currentItems;
	ContentBrowserSelectionStack selectedItems;
	ContentBrowserSelectionStack copiedAssets;
	ContentBrowserDirectory* baseDirectory;
	ContentBrowserDirectory* previousDirectory;
	ContentBrowserDirectory* currentDirectory;
	ContentBrowserDirectory* nextDirectory;
	bool m_UpdateNavigationPath = false;
	std::vector<ContentBrowserDirectory*> m_BreadCrumbData;
	std::unordered_map<UUID, ContentBrowserDirectory*> directories;
	SharedPtr<Texture> fileIcon;
	SharedPtr<Texture> directoryIcon;
	SharedPtr<Texture> backButtonIcon;
	SharedPtr<Texture> forwardButtonIcon;
	SharedPtr<Texture> refreshButtonIcon;
	std::map<std::string, SharedPtr<Texture>> AssetIconMap;
	float thumbnailSize = 96.0f;
	float padding = 2.0f;
	float searchBarWidth = 200.0f;
	char renameBuffer[MAX_INPUT_BUFFER_LENGTH];
	char searchBuffer[MAX_INPUT_BUFFER_LENGTH];
};

}