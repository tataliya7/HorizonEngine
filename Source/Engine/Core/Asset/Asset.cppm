module;

#include <string>
#include <memory>
#include <filesystem>
#include <unordered_map>

export module HorizonEngine.Core.Asset;

import HorizonEngine.Core.Types;
import HorizonEngine.Core.Math;

export namespace HE
{
	struct Asset
	{
		std::string filename;
	};

	class AssetImporter
	{
	public:
		virtual void ImportAsset(const char* file) = 0;
	};

	class AssetManager
	{
	public:
		static void AddAsset(const std::string& path, Asset* asset)
		{
			ImportedAssets.emplace(path, asset);
		}
		template<typename T>
		static T* GetAsset(const std::string& assetHandle)
		{
			if (ImportedAssets.find(assetHandle) == ImportedAssets.end())
			{
				return nullptr;
			}
			return (T*)(ImportedAssets[assetHandle].get());
		}
		static std::unordered_map<std::string, std::shared_ptr<Asset>> ImportedAssets;
	};

	enum MaterialFlags
	{
		MATERIAL_FLAGS_NONE = 0x00000000,
		MATERIAL_FLAGS_USE_BASE_COLOR_MAP = 0x00000001,
		MATERIAL_FLAGS_USE_NORMAL_MAP = 0x00000002,
		MATERIAL_FLAGS_USE_METALLIC_ROUGHNESS_MAP = 0x00000004,
		MATERIAL_FLAGS_USE_EMISSIVE_MAP = 0x00000008,
		MATERIAL_FLAGS_MASK_ALL = 0xffffffff,
	};

	struct Material
	{
		std::string name;
		int32 flags;
		Vector4 baseColor;
		float metallic;
		float specular;
		float roughness;
		Vector4 emission;
		float emissionStrength;
		float alpha;
		std::string baseColorMap;
		std::string normalMap;
		std::string metallicRoughnessMap;
		std::string emissiveMap;
	};

	struct MeshElement
	{
		std::string name;
		uint32 baseVertex;
		uint32 baseIndex;
		uint32 materialIndex;
		uint32 numIndices;
		uint32 numVertices;
		Matrix4x4 transform;
	};

	class Mesh : public Asset
	{
	public:

		uint32 numVertices;
		uint32 numIndices;

		std::vector<Vector3> positions;
		std::vector<Vector3> normals;
		std::vector<Vector4> tangents;
		std::vector<Vector2> texCoords;
		std::vector<uint32> indices;

		std::vector<Material> materials;
		std::vector<MeshElement> elements;
	};
}

namespace HE
{
	std::unordered_map<std::string, std::shared_ptr<Asset>> AssetManager::ImportedAssets;
}