#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>
#include <assimp/Exceptional.h>

#include <map>

#include "Core/CoreDefinitions.h"
#include "AssimpImporter.h"

namespace HE
{
	Matrix4x4 ConvertToMatrix4x4(const aiMatrix4x4& matrix)
	{
		Matrix4x4 result;
		result[0][0] = matrix.a1; result[1][0] = matrix.a2; result[2][0] = matrix.a3; result[3][0] = matrix.a4;
		result[0][1] = matrix.b1; result[1][1] = matrix.b2; result[2][1] = matrix.b3; result[3][1] = matrix.b4;
		result[0][2] = matrix.c1; result[1][2] = matrix.c2; result[2][2] = matrix.c3; result[3][2] = matrix.c4;
		result[0][3] = matrix.d1; result[1][3] = matrix.d2; result[2][3] = matrix.d3; result[3][3] = matrix.d4;
		return result;
	}

	struct ImportAssimpTaskData
	{
		const char* filename;
		Mesh* mesh;
	};

	static void ImportAssimpNode(Mesh* mesh, const struct aiNode* aiNode, const Matrix4x4& parentTransform)
	{
		Matrix4x4 transform = parentTransform * ConvertToMatrix4x4(aiNode->mTransformation);
		for (uint32 i = 0; i < aiNode->mNumMeshes; i++)
		{
			int meshIndex = aiNode->mMeshes[i];
			auto& element = mesh->elements[meshIndex];
			element.transform = transform;
		}

		for (uint32 i = 0; i < aiNode->mNumChildren; i++)
		{
			ImportAssimpNode(mesh, aiNode->mChildren[i], transform);
		}
	}

	static void ImportAssimpTask(void* data)
	{
		ImportAssimpTaskData* taskData = (ImportAssimpTaskData*)data;

		HE_LOG_INFO("Import scene: {0}", taskData->filename);

		static const uint32_t meshImportFlags = aiProcessPreset_TargetRealtime_MaxQuality;
		//static const uint32_t meshImportFlags =
		//	aiProcess_CalcTangentSpace |        // Create binormals/tangents just in case
		//	aiProcess_Triangulate |             // Make sure we're triangles
		//	aiProcess_SortByPType |             // Split meshes by primitive type
		//	aiProcess_GenNormals |              // Make sure we have legit normals
		//	aiProcess_GenUVCoords |             // Convert UVs if required 
		//	aiProcess_OptimizeMeshes |          // Batch draws where possible
		//	aiProcess_JoinIdenticalVertices |
		//	aiProcess_ValidateDataStructure;    // Validation

		std::unique_ptr<Assimp::Importer> importer = std::make_unique<Assimp::Importer>();
		const aiScene* aiScene = importer->ReadFile(taskData->filename, meshImportFlags);

		std::string dir = std::string(taskData->filename).substr(0, std::string(taskData->filename).find_last_of('/'));

		Mesh* mesh = taskData->mesh;

		if (!aiScene)
		{
			HE_LOG_INFO("Failed to load mesh file: {0}", taskData->filename);
		}

		if (aiScene->HasMaterials())
		{
			for (uint32 i = 0; i < aiScene->mNumMaterials; i++)
			{
				const aiMaterial* aiMaterial = aiScene->mMaterials[i];

				Material& material = mesh->materials.emplace_back();
				material.name = aiMaterial->GetName().C_Str();

				Vector4 baseColor = Vector4(1.0f, 1.0f, 1.0f, 1.0f);
				aiColor3D aiColor;
				if (aiMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, aiColor) == AI_SUCCESS)
				{
					material.baseColor = { aiColor.r, aiColor.g, aiColor.b, 1.0f };
				}

				float metallic = 0.0f;
				aiMaterial->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
				material.metallic = metallic;

				float roughness = 1.0f;
				aiMaterial->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
				material.roughness = roughness;

				Vector4 emission = Vector4(0.0f, 0.0f, 0.0f, 0.0f);
				if (aiMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, aiColor) == AI_SUCCESS)
				{
					material.emission = { aiColor.r, aiColor.g, aiColor.b, 0.0f };
				}

				aiString aiTexPath;

				aiReturn result = aiMaterial->GetTexture(aiTextureType_BASE_COLOR, 0, &aiTexPath);
				if (result == aiReturn_SUCCESS)
				{
					material.baseColorMap = dir + '/' + aiTexPath.C_Str();
					material.flags |= MATERIAL_FLAGS_USE_BASE_COLOR_MAP;
				}
				else
				{
					material.baseColorMap = "";
				}

				result = aiMaterial->GetTexture(aiTextureType_NORMALS, 0, &aiTexPath);
				if (result == aiReturn_SUCCESS)
				{
					material.normalMap = dir + '/' + aiTexPath.C_Str();
					// material.flags |= MATERIAL_FLAGS_USE_NORMAL_MAP;
				}
				else
				{
					material.normalMap = "";
				}

				result = aiMaterial->GetTexture(aiTextureType_UNKNOWN, 0, &aiTexPath);
				if (result == aiReturn_SUCCESS)
				{
					material.metallicRoughnessMap = dir + '/' + aiTexPath.C_Str();
					material.flags |= MATERIAL_FLAGS_USE_METALLIC_ROUGHNESS_MAP;
				}
				else
				{
					material.metallicRoughnessMap = "";
				}

				result = aiMaterial->GetTexture(aiTextureType_EMISSIVE, 0, &aiTexPath);
				if (result == aiReturn_SUCCESS)
				{
					material.emissiveMap = dir + '/' + aiTexPath.C_Str();
					material.flags |= MATERIAL_FLAGS_USE_EMISSIVE_MAP;
				}
				else
				{
					material.emissiveMap = "";
				}
			}
		}

		mesh->numVertices = 0;
		mesh->numIndices = 0;

		if (aiScene->HasMeshes())
		{
			for (uint32 i = 0; i < aiScene->mNumMeshes; i++)
			{
				const aiMesh* aiMesh = aiScene->mMeshes[i];

				if (!aiMesh->HasPositions() || !aiMesh->HasNormals())
				{
					HE_LOG_ERROR("Failed to import mesh {}.", i);
					return;
				}

				MeshElement& element = mesh->elements.emplace_back();
				element.name = aiMesh->mName.C_Str();
				element.baseVertex = mesh->numVertices;
				element.baseIndex = mesh->numIndices;
				element.materialIndex = aiMesh->mMaterialIndex;
				element.numVertices = aiMesh->mNumVertices;
				element.numIndices = aiMesh->mNumFaces * 3;
				element.transform = Matrix4x4(1.0f);

				for (uint32 vertexID = 0; vertexID < aiMesh->mNumVertices; vertexID++)
				{
					Vector3 normal = Vector3(aiMesh->mNormals[vertexID].x, aiMesh->mNormals[vertexID].y, aiMesh->mNormals[vertexID].z);
					mesh->positions.emplace_back(Vector3(aiMesh->mVertices[vertexID].x, aiMesh->mVertices[vertexID].y, aiMesh->mVertices[vertexID].z));
					mesh->normals.emplace_back(normal);
					if (aiMesh->HasTextureCoords(0))
					{
						mesh->texCoords.emplace_back(Vector2(aiMesh->mTextureCoords[0][vertexID].x, aiMesh->mTextureCoords[0][vertexID].y));
					}
					else
					{
						mesh->texCoords.emplace_back(Vector2(0.0f));
					}

					if (aiMesh->HasTangentsAndBitangents())
					{
						Vector3 tangent = Vector3(aiMesh->mTangents[vertexID].x, aiMesh->mTangents[vertexID].y, aiMesh->mTangents[vertexID].z);
						Vector3 bitangent = Vector3(aiMesh->mBitangents[vertexID].x, aiMesh->mBitangents[vertexID].y, aiMesh->mBitangents[vertexID].z);
						float tangentW = glm::dot(glm::cross(normal, tangent), bitangent) > 0.0f ? 1.0f : -1.0f;
						mesh->tangents.emplace_back(Vector4(aiMesh->mTangents[vertexID].x, aiMesh->mTangents[vertexID].y, aiMesh->mTangents[vertexID].z, tangentW));
					}
					else
					{
						mesh->tangents.emplace_back(Vector4(0.0f));
					}
				}

				for (uint32 faceIndex = 0; faceIndex < aiMesh->mNumFaces; faceIndex++)
				{
					ASSERT(aiMesh->mFaces[faceIndex].mNumIndices == 3);
					mesh->indices.emplace_back(aiMesh->mFaces[faceIndex].mIndices[0] + element.baseVertex);
					mesh->indices.emplace_back(aiMesh->mFaces[faceIndex].mIndices[1] + element.baseVertex);
					mesh->indices.emplace_back(aiMesh->mFaces[faceIndex].mIndices[2] + element.baseVertex);
				}

				mesh->numVertices += element.numVertices;
				mesh->numIndices += element.numIndices;
			}
		}

		static Quaternion zUpQuat = glm::rotate(glm::quat(), Math::DegreesToRadians(90.0), Vector3(1.0, 0.0, 0.0));
		static Matrix4x4 preTransform = Math::Compose(Vector3(0.0f, 0.0f, 0.0f), zUpQuat, Vector3(1.0f, 1.0f, 1.0f));
		ImportAssimpNode(mesh, aiScene->mRootNode, preTransform);
	}

	void AssimpImporter::ImportAsset(const char* filename)
	{
		Mesh* mesh = new Mesh();
		ImportAssimpTaskData taskData = {
			.filename = filename,
			.mesh = mesh,
		};
		ImportAssimpTask(&taskData);
		AssetManager::AddAsset(filename, mesh);
	}

	bool ImportOBJ(const char* filename, OBJImportSettings settings)
	{
		return true;
	}

	bool ImportFBX(const char* filename, FBXImportSettings settings)
	{
		return true;
	}

	bool ImportGLTF2(const char* filename, GLTF2ImportSettings settings)
	{
		return true;
	}
}