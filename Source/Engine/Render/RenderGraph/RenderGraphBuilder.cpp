module HorizonEngine.Render.RenderGraph;

namespace HE
{
	RenderGraphTextureHandle RenderGraphBuilder::CreateTransientTexture(const RenderGraphTextureDesc& desc, const char* name)
	{
		return RenderGraphTextureHandle();
	}

	RenderGraphBufferHandle RenderGraphBuilder::CreateTransientBuffer(const RenderGraphBufferDesc& desc, const char* name)
	{
		return RenderGraphBufferHandle();
	}

	RenderGraphTextureHandle RenderGraphBuilder::ReadTexture(RenderGraphTextureHandle handle, RenderBackendResourceState finalState, const RenderGraphTextureSubresourceRange& range)
	{
		pass->textureStates.push_back(RenderGraphPass::TextureState{
			.texture = renderGraph->textures[handle.GetIndex()],
			.state = finalState,
			.subresourceRange = range,
		});
		pass->inputs.push_back(renderGraph->textures[handle.GetIndex()]);
		renderGraph->textures[handle.GetIndex()]->refCount++;
		return handle;
	}

	RenderGraphTextureHandle RenderGraphBuilder::WriteTexture(RenderGraphTextureHandle handle, RenderBackendResourceState finalState, const RenderGraphTextureSubresourceRange& range)
	{
		pass->textureStates.push_back(RenderGraphPass::TextureState{
			.texture = renderGraph->textures[handle.GetIndex()],
			.state = finalState,
			.subresourceRange = range,
		});
		pass->outputs.push_back(renderGraph->textures[handle.GetIndex()]);
		pass->refCount++;
		return handle;
	}

	RenderGraphTextureHandle RenderGraphBuilder::ReadWriteTexture(RenderGraphTextureHandle handle, RenderBackendResourceState finalState, const RenderGraphTextureSubresourceRange& range)
	{
		pass->textureStates.push_back(RenderGraphPass::TextureState{
			.texture = renderGraph->textures[handle.GetIndex()],
			.state = finalState,
			.subresourceRange = range,
		});
		pass->inputs.push_back(renderGraph->textures[handle.GetIndex()]);
		pass->outputs.push_back(renderGraph->textures[handle.GetIndex()]);
		pass->refCount++;
		return handle;
	}

	RenderGraphBufferHandle RenderGraphBuilder::ReadBuffer(RenderGraphBufferHandle handle, RenderBackendResourceState initalState)
	{
		return RenderGraphBufferHandle();
	}

	RenderGraphBufferHandle RenderGraphBuilder::WriteBuffer(RenderGraphBufferHandle handle, RenderBackendResourceState initalState)
	{
		return RenderGraphBufferHandle();
	}

	RenderGraphBufferHandle RenderGraphBuilder::ReadWriteBuffer(RenderGraphBufferHandle handle, RenderBackendResourceState initalState)
	{
		return RenderGraphBufferHandle();
	}

	void RenderGraphBuilder::BindColorTarget(uint32 slot, RenderGraphTextureHandle handle, RenderTargetLoadOp loadOp, RenderTargetStoreOp storeOp, uint32 mipLevel, uint32 arraylayer)
	{
		pass->colorTargets[slot] = {
			.texture = handle,
			.mipLevel = mipLevel,
			.arrayLayer = arraylayer,
			.loadOp = loadOp,
			.storeOp = storeOp,
		};
	}

	void RenderGraphBuilder::BindDepthTarget(RenderGraphTextureHandle handle,  RenderTargetLoadOp depthLoadOp, RenderTargetStoreOp depthStoreOp, uint32 mipLevel, uint32 arraylayer)
	{
		pass->depthStentcilTarget = {
			.texture = handle,
			.mipLevel = mipLevel,
			.arrayLayer = arraylayer,
			.depthLoadOp = depthLoadOp,
			.depthStoreOp = depthStoreOp,
			.stencilLoadOp = RenderTargetLoadOp::DontCare,
			.stencilStoreOp = RenderTargetStoreOp::DontCare,
		};
	}

	void RenderGraphBuilder::BindDepthStencilTarget(RenderGraphTextureHandle handle, RenderTargetLoadOp depthLoadOp, RenderTargetStoreOp depthStoreOp, RenderTargetLoadOp stencilLoadOp, RenderTargetStoreOp stencilStoreOp, uint32 mipLevel, uint32 arraylayer)
	{
		pass->depthStentcilTarget = {
			.texture = handle,
			.mipLevel = mipLevel,
			.arrayLayer = arraylayer,
			.depthLoadOp = depthLoadOp,
			.depthStoreOp = depthStoreOp,
			.stencilLoadOp = stencilLoadOp,
			.stencilStoreOp = stencilStoreOp,
		};
	}
}