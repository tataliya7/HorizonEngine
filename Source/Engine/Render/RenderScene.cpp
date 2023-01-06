module;

#include <ECS/ECS.h>
#include <Daisy/DaisyRenderer.h>
#include "AssimpImporter/AssimpImporter.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

module HorizonEngine.Render.Scene;

import HorizonEngine.Render.Enviroment;
import HorizonEngine.Render.ShaderSystem;
import HorizonEngine.Physics;

namespace HE
{
	RenderScene::RenderScene()
	{

	}

	RenderScene::~RenderScene()
	{

	}

	void RenderScene::Update(float deltaTime)
	{
		scene->GetEntityManager()->GetView<StaticMeshComponent>().each([&](EntityHandle entity, auto& component)
		{
			auto& transformComponent = scene->GetEntityManager()->GetComponent<TransformComponent>(entity);
			component.proxy->worldMatrix = transformComponent.world;
		});

		for (auto& mesh : meshes)
		{
			for (auto& renderable : mesh->renderables)
			{
				worldMatrices[renderable.transformIndex] = renderable.localMatrix * mesh->worldMatrix;
			}
		}

		void* data = nullptr;
		RenderBackendMapBuffer(renderBackend, worldMatrixBuffer, &data);
		memcpy((uint8*)data, worldMatrices.data(), worldMatrices.size() * sizeof(Matrix4x4));
		RenderBackendUnmapBuffer(renderBackend, worldMatrixBuffer);
	}

	RenderBackendTextureHandle LoadTextureFromHDRFile(RenderBackend* renderBackend, const char* filename)
	{
		if (!stbi_is_hdr(filename))
		{
			return RenderBackendTextureHandle::NullHandle;
		}

		int iw = 0, ih = 0, c = 0;
		stbi_set_flip_vertically_on_load(false);
		void* data = stbi_loadf(filename, &iw, &ih, &c, STBI_rgb_alpha);
		if (data == nullptr)
		{
			return RenderBackendTextureHandle::NullHandle;
		}
		uint64 bufferSize = iw * ih * 4 * sizeof(float);

		RenderBackendTextureDesc desc = RenderBackendTextureDesc::CreateTexture2D(iw, ih, 1, PixelFormat::RGBA32Float);
		RenderBackendTextureHandle texture = RenderBackendCreateTexture(renderBackend, ~0u, &desc, data, filename);

		stbi_image_free(data);
		return texture;
	}

	RenderBackendTextureHandle LoadTextureFromFile(RenderBackend* renderBackend, const char* filename, bool autoMipmaps = true, bool filpY = true)
	{
		int iw = 0, ih = 0, c = 0;
		stbi_set_flip_vertically_on_load(filpY);
		unsigned char* data = stbi_load(filename, &iw, &ih, &c, STBI_default);
		if (data == nullptr)
		{
			return RenderBackendTextureHandle::NullHandle;
		}
		uint64 bufferSize = iw * ih * 4;
		unsigned char* buffer = (unsigned char*)_aligned_malloc(bufferSize, 32);

		for (uint32 y = 0; y < (uint32)ih; y++)
		{
			for (uint32 x = 0; x < (uint32)iw; x++)
			{
				uint32 idx = x + y * iw;
				switch (c)
				{
				case STBI_grey:
				{
					buffer[idx * 4 + 0] = data[idx];
					buffer[idx * 4 + 2] = buffer[idx * 4 + 1] = buffer[idx * 4 + 0];
					buffer[idx * 4 + 3] = 255;
					break;
				}
				case STBI_grey_alpha:
				{
					buffer[idx * 4 + 0] = data[idx * 2 + 0];
					buffer[idx * 4 + 2] = buffer[idx * 4 + 1] = buffer[idx * 4 + 0];
					buffer[idx * 4 + 3] = data[idx * 2 + 1];
					break;
				}
				case STBI_rgb:
				{
					buffer[idx * 4 + 0] = data[idx * 3 + 0];
					buffer[idx * 4 + 1] = data[idx * 3 + 1];
					buffer[idx * 4 + 2] = data[idx * 3 + 2];
					buffer[idx * 4 + 3] = 255;
					break;
				}
				case STBI_rgb_alpha:
				{
					buffer[idx * 4 + 0] = data[idx * 4 + 0];
					buffer[idx * 4 + 1] = data[idx * 4 + 1];
					buffer[idx * 4 + 2] = data[idx * 4 + 2];
					buffer[idx * 4 + 3] = data[idx * 4 + 3];
					break;
				}
				default: break;
				}
			}
		}

		stbi_image_free(data);

		RenderBackendTextureDesc desc = RenderBackendTextureDesc::CreateTexture2D(iw, ih, autoMipmaps ? Math::MaxNumMipLevels(iw, ih) : 1, PixelFormat::RGBA8Unorm);
		RenderBackendTextureHandle texture = RenderBackendCreateTexture(renderBackend, ~0u, &desc, buffer, filename);

		_aligned_free(buffer);
		
		return texture;
	}

	void EquirectangularToCubemap(RenderCommandList& commandList, RenderBackendTextureHandle equirectangular, RenderBackendTextureHandle cubemap, uint32 cubemapSize)
	{
		ShaderLibrary* shaderLibrary = GetGlobalShaderLibrary();
		RenderBackendShaderHandle computeShader = shaderLibrary->GetShader("EquirectangularToCubemapCS");

		RenderBackendBarrier transition(cubemap, RenderBackendTextureSubresourceRange(0, 1, 0, 6), RenderBackendResourceState::Undefined, RenderBackendResourceState::UnorderedAccess);
		commandList.Transitions(&transition, 1);

		uint32 dispatchX = CEIL_DIV(cubemapSize, 8);
		uint32 dispatchY = CEIL_DIV(cubemapSize, 8);
		uint32 dispatchZ = 1;

		ShaderArguments shaderArguments = {};
		shaderArguments.BindTextureSRV(0, RenderBackendTextureSRVDesc::Create(equirectangular));
		shaderArguments.BindTextureUAV(1, RenderBackendTextureUAVDesc::Create(cubemap, 0));

		commandList.Dispatch(
			computeShader,
			shaderArguments,
			dispatchX,
			dispatchY,
			dispatchZ);
	}

	void RenderScene::SetSkyLight(SkyLightRenderProxy* proxy)
	{
		ASSERT(proxy);

		RenderScene* scene = this;
		{
			scene->skyLight = proxy;
		}
	}

	void RenderScene::UpdateSkyLights()
	{
		//uint32 cubemapSize = skyLightComponent->CubemapResolution;
		//EquirectangularToCubemap();
		//ComputeEnviromentCubemaps(commandList, environmentMap, cubemapSize, outIrradianceEnvironmentMap, outFilteredEnvironmentMap);
	}

	void RenderScene::Setup(Scene* scene, DaisyRenderer* renderer)
	{
		Setup(scene, renderer->GetRenderBackend(), renderer->arena);
	}

	void RenderScene::Setup(Scene* scene, RenderBackend* renderBackend, MemoryArena* arena)
	{
		this->renderBackend = renderBackend;

		uint32 deviceMask = ~0u;

		this->scene = scene;

		scene->GetEntityManager()->GetView<TransformComponent>().each([&](EntityHandle entity, auto& component)
		{
			component.Update();
		});
	
		scene->GetEntityManager()->GetView<BoxColliderComponent>().each([&](EntityHandle entity, auto& component)
		{
			scene->physicsScene->CreateActor(scene->GetEntityManager(), entity);
		});

		scene->GetEntityManager()->GetView<DirectionalLightComponent>().each([&](EntityHandle entity, auto& component)
		{
			component.proxy = new LightRenderProxy(&component);
			simpleDirectionalLight = component.proxy;
		});

		scene->GetEntityManager()->GetView<SkyLightComponent>().each([&](EntityHandle entity, auto& component)
		{
			component.proxy = new SkyLightRenderProxy(&component);
			SetSkyLight(component.proxy);
		});

		auto group = scene->GetEntityManager()->Get()->group<StaticMeshComponent>(entt::get<TransformComponent>);
		for (auto entity : group)
		{
			auto [transformComponent, staticMeshComponent] = group.get<TransformComponent, StaticMeshComponent>(entity);

			AssimpImporter assimpImporter;
			assimpImporter.ImportAsset(staticMeshComponent.meshSource.c_str());

			Mesh* meshSource = AssetManager::GetAsset<Mesh>(staticMeshComponent.meshSource);

			MeshRenderProxy* proxy = new MeshRenderProxy(&staticMeshComponent);
			proxy->worldMatrix = transformComponent.world;
			staticMeshComponent.proxy = proxy;
			meshes.push_back(proxy);

			if (meshSource)
			{
				RenderBackendBufferDesc vertexBuffer0Desc = RenderBackendBufferDesc::CreateByteAddress(meshSource->numVertices * sizeof(Vector3));
				RenderBackendBufferHandle vertexBuffer0 = RenderBackendCreateBuffer(renderBackend, deviceMask, &vertexBuffer0Desc, "VertexPosition");

				void* data = nullptr;
				RenderBackendMapBuffer(renderBackend, vertexBuffer0, &data);
				memcpy((uint8*)data, meshSource->positions.data(), meshSource->numVertices * sizeof(Vector3));
				RenderBackendUnmapBuffer(renderBackend, vertexBuffer0);

				RenderBackendBufferDesc vertexBuffer1Desc = RenderBackendBufferDesc::CreateByteAddress(meshSource->numVertices * sizeof(Vector3));
				RenderBackendBufferHandle vertexBuffer1 = RenderBackendCreateBuffer(renderBackend, deviceMask, &vertexBuffer1Desc, "VertexNormal");
		
				RenderBackendMapBuffer(renderBackend, vertexBuffer1, &data);
				memcpy((uint8*)data, meshSource->normals.data(), meshSource->numVertices * sizeof(Vector3));
				RenderBackendUnmapBuffer(renderBackend, vertexBuffer1);

				RenderBackendBufferDesc vertexBuffer2Desc = RenderBackendBufferDesc::CreateByteAddress(meshSource->numVertices * sizeof(Vector4));
				RenderBackendBufferHandle vertexBuffer2 = RenderBackendCreateBuffer(renderBackend, deviceMask, &vertexBuffer2Desc, "VertexTangent");

				RenderBackendMapBuffer(renderBackend, vertexBuffer2, &data);
				memcpy((uint8*)data, meshSource->tangents.data(), meshSource->numVertices * sizeof(Vector4));
				RenderBackendUnmapBuffer(renderBackend, vertexBuffer2);

				RenderBackendBufferDesc vertexBuffer3Desc = RenderBackendBufferDesc::CreateByteAddress(meshSource->numVertices * sizeof(Vector2));
				RenderBackendBufferHandle vertexBuffer3 = RenderBackendCreateBuffer(renderBackend, deviceMask, &vertexBuffer3Desc, "VertexTexcoord");

				RenderBackendMapBuffer(renderBackend, vertexBuffer3, &data);
				memcpy((uint8*)data, meshSource->texCoords.data(), meshSource->numVertices * sizeof(Vector2));
				RenderBackendUnmapBuffer(renderBackend, vertexBuffer3);

				RenderBackendBufferDesc indexBufferDesc = RenderBackendBufferDesc::CreateByteAddress(meshSource->numIndices * sizeof(uint32));
				RenderBackendBufferHandle indexBuffer = RenderBackendCreateBuffer(renderBackend, deviceMask, &indexBufferDesc, "IndexBuffer");

				RenderBackendMapBuffer(renderBackend, indexBuffer, &data);
				memcpy((uint8*)data, meshSource->indices.data(), meshSource->numIndices * sizeof(uint32));
				RenderBackendUnmapBuffer(renderBackend, indexBuffer);

				uint32 vertexBufferIndex = (uint32)vertexBuffers[0].size();
				uint32 indexBufferIndex = (uint32)indexBuffers.size();

				vertexBuffers[0].push_back(vertexBuffer0);
				vertexBuffers[1].push_back(vertexBuffer1);
				vertexBuffers[2].push_back(vertexBuffer2);
				vertexBuffers[3].push_back(vertexBuffer3);
				indexBuffers.push_back(indexBuffer);

				uint32 baseMaterialIndex = (uint32)materials.size();
				for (uint32 i = 0; i < (uint32)meshSource->materials.size(); i++)
				{
					PBRMaterialShaderParameters& material = materials.emplace_back();

					material.flags = meshSource->materials[i].flags;
					material.baseColor = meshSource->materials[i].baseColor;
					material.metallic = meshSource->materials[i].metallic;
					material.roughness = meshSource->materials[i].roughness;
					material.emission = meshSource->materials[i].emission;
					
					if (meshSource->materials[i].baseColorMap == "")
					{
						material.baseColorMapIndex = 0;
					}
					else
					{
						RenderBackendTextureHandle texture = LoadTextureFromFile(renderBackend, meshSource->materials[i].baseColorMap.c_str());
						textures.push_back(texture);
						material.baseColorMapIndex = RenderBackendGetTextureSRVDescriptorIndex(renderBackend, deviceMask, texture);
					}
					if (meshSource->materials[i].normalMap == "")
					{
						material.normalMapIndex = 0;
					}
					else
					{
						RenderBackendTextureHandle texture = LoadTextureFromFile(renderBackend, meshSource->materials[i].normalMap.c_str());
						textures.push_back(texture);
						material.normalMapIndex = RenderBackendGetTextureSRVDescriptorIndex(renderBackend, deviceMask, texture);
					}
					if (meshSource->materials[i].metallicRoughnessMap == "")
					{
						material.metallicRoughnessMapIndex = 0;
					}
					else
					{
						RenderBackendTextureHandle texture = LoadTextureFromFile(renderBackend, meshSource->materials[i].metallicRoughnessMap.c_str());
						textures.push_back(texture);
						material.metallicRoughnessMapIndex = RenderBackendGetTextureSRVDescriptorIndex(renderBackend, deviceMask, texture);
					}
					if (meshSource->materials[i].emissiveMap == "")
					{
						material.emissiveMapIndex = 0;
					}
					else
					{
						RenderBackendTextureHandle texture = LoadTextureFromFile(renderBackend, meshSource->materials[i].emissiveMap.c_str());
						textures.push_back(texture);
						material.emissiveMapIndex = RenderBackendGetTextureSRVDescriptorIndex(renderBackend, deviceMask, texture);
					}
				}

				for (const auto& element : meshSource->elements)
				{
					uint32 transformIndex = (uint32)worldMatrices.size();
					worldMatrices.push_back(element.transform * transformComponent.world);
					auto& renderable = renderables.emplace_back();
					renderable.localMatrix = element.transform;
					renderable.firstVertex = element.baseVertex;
					renderable.firstIndex = element.baseIndex;
					renderable.numIndices = element.numIndices;
					renderable.numVertices = element.numVertices;
					renderable.vertexBufferIndex = vertexBufferIndex;
					renderable.indexBufferIndex = indexBufferIndex;
					renderable.materialIndex = baseMaterialIndex + element.materialIndex;
					renderable.transformIndex = transformIndex;

					proxy->renderables.push_back(renderable);
				}
			}
		}

		RenderBackendBufferDesc materialBufferDesc = RenderBackendBufferDesc::CreateByteAddress(materials.size() * sizeof(PBRMaterialShaderParameters));
		materialBuffer = RenderBackendCreateBuffer(renderBackend, deviceMask, &materialBufferDesc, "MaterialBuffer");

		void* data = nullptr;
		RenderBackendMapBuffer(renderBackend, materialBuffer, &data);
		memcpy((uint8*)data, materials.data(), materials.size() * sizeof(PBRMaterialShaderParameters));
		RenderBackendUnmapBuffer(renderBackend, materialBuffer);

		RenderBackendBufferDesc worldMatrixBufferDesc = RenderBackendBufferDesc::CreateByteAddress(worldMatrices.size() * sizeof(Matrix4x4));
		worldMatrixBuffer = RenderBackendCreateBuffer(renderBackend, deviceMask, &worldMatrixBufferDesc, "WorldMatrixBuffer");

		RenderBackendMapBuffer(renderBackend, worldMatrixBuffer, &data);
		memcpy((uint8*)data, worldMatrices.data(), worldMatrices.size() * sizeof(Matrix4x4));
		RenderBackendUnmapBuffer(renderBackend, worldMatrixBuffer);

#if DEBUG_ONLY_RAY_TRACING_ENBALE

		for (auto& transform : worldMatrices)
		{
			transform = Math::Transpose(transform);
		}
		worldMatrixBuffer1 = RenderBackendCreateBuffer(renderBackend, deviceMask, &worldMatrixBufferDesc, "WorldMatrixBuffer");
		RenderBackendMapBuffer(renderBackend, worldMatrixBuffer1, &data);
		memcpy((uint8*)data, worldMatrices.data(), worldMatrices.size() * sizeof(Matrix4x4));
		RenderBackendUnmapBuffer(renderBackend, worldMatrixBuffer1);

		std::vector<RenderBackendGeometryDesc> geometryDescs(renderables.size());
		for (uint32 i = 0; i < (uint32)geometryDescs.size(); i++)
		{
			geometryDescs[i] = {
				.type = RenderBackendGeometryType::Triangles, 
				.flags = RenderBackendGeometryFlags::Opaque, 
				.triangleDesc = {
					.numIndices = renderables[i].numIndices,
					.numVertices = renderables[i].numVertices,
					.vertexStride = 3 * sizeof(float),
					.vertexBuffer = vertexBuffers[0][renderables[i].vertexBufferIndex],
					.vertexOffset = 0,
					.indexBuffer = indexBuffers[renderables[i].indexBufferIndex],
					.indexOffset = renderables[i].firstIndex * sizeof(uint32),
					.transformBuffer = worldMatrixBuffer1,
					.transformOffset = renderables[i].transformIndex * 16 * sizeof(float),
				}
			};
		}
		
		RenderBackendBottomLevelASDesc bottomLevelASDesc = {
			.buildFlags = RenderBackendAccelerationStructureBuildFlags::PreferFastTrace,
			.numGeometries = (uint32)geometryDescs.size(),
			.geometryDescs = geometryDescs.data(),
		};
		bottomLevelAS = RenderBackendCreateBottomLevelAS(renderBackend, deviceMask, &bottomLevelASDesc, "BottomLevelAS");

		RenderBackendRayTracingInstance geometryInstance = {
			.transformMatrix = Matrix4x4(1.0),
			.instanceID = 0,
			.instanceMask = 0xff,
			.instanceContributionToHitGroupIndex = 0,
			.flags = RenderBackendRayTracingInstanceFlags::TriangleFacingCullDisable,
			.blas = bottomLevelAS,
		};

		RenderBackendTopLevelASDesc topLevelASDesc = {
			.buildFlags = RenderBackendAccelerationStructureBuildFlags::PreferFastTrace,
			.geometryFlags = RenderBackendGeometryFlags::Opaque,
			.numInstances = 1,
			.instances = &geometryInstance,
		};
		topLevelAS = RenderBackendCreateTopLevelAS(renderBackend, deviceMask, &topLevelASDesc, "TopLevelAS");
#endif

		uint32 cubemapSize = skyLight->component->cubemapResolution;
		RenderBackendTextureHandle equirectangular = LoadTextureFromHDRFile(renderBackend, skyLight->component->cubemap.c_str());
		RenderBackendTextureDesc cubemapDesc = RenderBackendTextureDesc::CreateCube(cubemapSize, PixelFormat::RGBA16Float, TextureCreateFlags::UnorderedAccess | TextureCreateFlags::ShaderResource, Math::MaxNumMipLevels(cubemapSize));
		skyLight->environmentMap = RenderBackendCreateTexture(renderBackend, deviceMask, &cubemapDesc, nullptr, "EnvironmentMap");
		
		RenderBackendTextureDesc irradianceEnvironmentMapDesc = RenderBackendTextureDesc::CreateCube(GIrradianceEnviromentMapSize, PixelFormat::RGBA16Float, TextureCreateFlags::UnorderedAccess | TextureCreateFlags::ShaderResource);
		skyLight->irradianceEnvironmentMap = RenderBackendCreateTexture(renderBackend, deviceMask, &irradianceEnvironmentMapDesc, nullptr, "IrradianceEnvironmentMap");
		
		skyLight->filteredEnvironmentMap = RenderBackendCreateTexture(renderBackend, deviceMask, &cubemapDesc, nullptr, "FilteredEnvironmentMap");

		RenderCommandList* commandList = new RenderCommandList(arena);

		EquirectangularToCubemap(*commandList, equirectangular, skyLight->environmentMap, cubemapSize);
		ComputeEnviromentCubemaps(*commandList, skyLight->environmentMap, cubemapSize, skyLight->irradianceEnvironmentMap, skyLight->filteredEnvironmentMap);

		RenderBackendSubmitRenderCommandLists(renderBackend, &commandList, 1);
	}
}