module HorizonEngine.Render.RenderGraph;

namespace HE
{
	RenderGraphTexture* RenderGraphRegistry::GetTexture(RenderGraphTextureHandle handle) const
	{
		return renderGraph->textures[handle.GetIndex()];
	}

	RenderGraphBuffer* RenderGraphRegistry::GetBuffer(RenderGraphBufferHandle handle) const
	{
		return renderGraph->buffers[handle.GetIndex()];
	}

	RenderGraphTexture* RenderGraphRegistry::GetImportedTexture(RenderBackendTextureHandle handle) const
	{
		if (renderGraph->importedTextures.find(handle) == renderGraph->importedTextures.end())
		{
			return nullptr;
		}
		return GetTexture(renderGraph->importedTextures[handle]);
	}

	RenderGraphBuffer* RenderGraphRegistry::GetImportedBuffer(RenderBackendBufferHandle handle) const
	{
		if (renderGraph->importedBuffers.find(handle) == renderGraph->importedBuffers.end())
		{
			return nullptr;
		}
		return GetBuffer(renderGraph->importedBuffers[handle]);
	}

	const RenderBackendTextureDesc& RenderGraphRegistry::GetTextureDesc(RenderGraphTextureHandle handle) const
	{
		return GetTexture(handle)->GetDesc();
	}

	const RenderBackendBufferDesc& RenderGraphRegistry::GetBufferDesc(RenderGraphBufferHandle handle) const
	{
		return GetBuffer(handle)->GetDesc();
	}

	RenderBackendTextureHandle RenderGraphRegistry::GetRenderBackendTexture(RenderGraphTextureHandle handle) const
	{
		return GetTexture(handle)->GetRenderBackendTexture();
	}

	RenderBackendBufferHandle RenderGraphRegistry::GetRenderBackendBuffer(RenderGraphBufferHandle handle) const
	{
		return GetBuffer(handle)->GetRenderBackendBuffer();
	}
}