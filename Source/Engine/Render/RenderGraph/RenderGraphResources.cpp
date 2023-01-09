module;

#include "Render/RenderDefinitions.h"

module HorizonEngine.Render.RenderGraph;

namespace HE
{
	const RenderGraphTextureSubresourceRange RenderGraphTextureSubresourceRange::WholeRange = RenderGraphTextureSubresourceRange(0, REMAINING_MIP_LEVELS, 0, REMAINING_ARRAY_LAYERS);

	RenderGraphResourcePool* GRenderGraphResourcePool = new RenderGraphResourcePool();

	void RenderGraphResourcePool::Tick()
	{
		for (auto& persistentTexture : allocatedTextures)
		{
			persistentTexture.active = false;
		}
		frameCounter++;
	}

	void RenderGraphResourcePool::CacheTexture(const RenderGraphPersistentTexture& texture)
	{
		for (auto& persistentTexture : allocatedTextures)
		{
			if (persistentTexture.texture == texture.texture)
			{
				return;
			}
		}

		RenderGraphPersistentTexture persistentTexture = {
			.active = false,
			.texture = texture.texture,
			.desc = texture.desc,
			.initialState = texture.initialState,
		};
		allocatedTextures.push_back(persistentTexture);
	}

	void RenderGraphResourcePool::ReleaseTexture(RenderBackendTextureHandle texture)
	{
		for (uint32 index = 0; index < allocatedTextures.size(); index++)
		{
			if (allocatedTextures[index].texture == texture)
			{
				allocatedTextures.erase(allocatedTextures.begin() + index);
				break;
			}
		}
	}

	RenderBackendTextureHandle RenderGraphResourcePool::FindOrCreateTexture(RenderBackend* backend, const RenderBackendTextureDesc* desc, const char* name)
	{
		for (auto& persistentTexture : allocatedTextures)
		{
			if (persistentTexture.active)
			{
				continue;
			}
			if (persistentTexture.desc == *desc)
			{
				persistentTexture.active = true;
				return persistentTexture.texture;
			}
		}

		uint32 deviceMask = ~0u;
		RenderBackendTextureHandle texture = RenderBackendCreateTexture(backend, deviceMask, desc, nullptr, name);
		RenderBackendResourceState initialState = RenderBackendResourceState::Undefined;

		RenderGraphPersistentTexture persistentTexture = {
			.active = true,
			.texture = texture,
			.desc = *desc,
			.initialState = initialState,
		};
		allocatedTextures.emplace_back(persistentTexture);
		return allocatedTextures.back().texture;
	}
}