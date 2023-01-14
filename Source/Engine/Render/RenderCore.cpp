module;

#include <fstream>

#include "Core/CoreDefinitions.h"

module HorizonEngine.Render.Core;

namespace HE
{
	RenderBackend* GRenderBackend = nullptr;

    const PixelFormatDesc GPixelFormatTable[] =
    {
        // format                         name                 type                           bytes  channels  { depth, stencil }   channelBits
        { PixelFormat::Unknown,           "Unknown",           PixelFormatType::Unknown,      0,     0,        { false, false },    { 0,  0,  0,  0  } },
                                                                                                           
        { PixelFormat::R8Unorm,           "R8Unorm",           PixelFormatType::Unorm,        1,     1,        { false, false },    { 8,  0,  0,  0  } },
        { PixelFormat::R8Snorm,           "R8Snorm",           PixelFormatType::Snorm,        1,     1,        { false, false },    { 8,  0,  0,  0  } },
        { PixelFormat::R16Unorm,          "R16Unorm",          PixelFormatType::Unorm,        2,     1,        { false, false },    { 16, 0,  0,  0  } },
        { PixelFormat::R16Snorm,          "R16Snorm",          PixelFormatType::Snorm,        2,     1,        { false, false },    { 16, 0,  0,  0  } },
        { PixelFormat::RG8Unorm,          "RG8Unorm",          PixelFormatType::Unorm,        2,     2,        { false, false },    { 8,  8,  0,  0  } },
        { PixelFormat::RG8Snorm,          "RG8Snorm",          PixelFormatType::Snorm,        2,     2,        { false, false },    { 8,  8,  0,  0  } },
        { PixelFormat::RG16Unorm,         "RG16Unorm",         PixelFormatType::Unorm,        4,     2,        { false, false },    { 16, 16, 0,  0  } },
        { PixelFormat::RG16Snorm,         "RG16Snorm",         PixelFormatType::Snorm,        4,     2,        { false, false },    { 16, 16, 0,  0  } },
        { PixelFormat::RGB16Unorm,        "RGB16Unorm",        PixelFormatType::Unorm,        6,     3,        { false, false },    { 16, 16, 16, 0  } },
        { PixelFormat::RGB16Snorm,        "RGB16Snorm",        PixelFormatType::Snorm,        6,     3,        { false, false },    { 16, 16, 16, 0  } },
        { PixelFormat::RGBA8Unorm,        "RGBA8Unorm",        PixelFormatType::Unorm,        4,     4,        { false, false },    { 8,  8,  8,  8  } },
        { PixelFormat::RGBA8Snorm,        "RGBA8Snorm",        PixelFormatType::Snorm,        4,     4,        { false, false },    { 8,  8,  8,  8  } },
        { PixelFormat::RGBA16Unorm,       "RGBA16Unorm",       PixelFormatType::Unorm,        8,     4,        { false, false },    { 16, 16, 16, 16 } },
                                                                                                         
        { PixelFormat::RGBA8UnormSrgb,    "RGBA8UnormSrgb",    PixelFormatType::UnormSrgb,    4,     4,        { false, false },    { 8,  8,  8,  8  } },
                                                                                                           
        { PixelFormat::R16Float,          "R16Float",          PixelFormatType::Float,        2,     1,        { false, false },    { 16, 0,  0,  0  } },
        { PixelFormat::RG16Float,         "RG16Float",         PixelFormatType::Float,        4,     2,        { false, false },    { 16, 16, 0,  0  } },
        { PixelFormat::RGB16Float,        "RGB16Float",        PixelFormatType::Float,        6,     3,        { false, false },    { 16, 16, 16, 0  } },
        { PixelFormat::RGBA16Float,       "RGBA16Float",       PixelFormatType::Float,        8,     4,        { false, false },    { 16, 16, 16, 16 } },
        { PixelFormat::R32Float,          "R32Float",          PixelFormatType::Float,        4,     1,        { false, false },    { 32, 0,  0,  0  } },
        { PixelFormat::RG32Float,         "RG32Float",         PixelFormatType::Float,        8,     2,        { false, false },    { 32, 32, 0,  0  } },
        { PixelFormat::RGB32Float,        "RGB32Float",        PixelFormatType::Float,        12,    3,        { false, false },    { 32, 32, 32, 0  } },
        { PixelFormat::RGBA32Float,       "RGBA32Float",       PixelFormatType::Float,        16,    4,        { false, false },    { 32, 32, 32, 32 } },
                                                                                                            
        { PixelFormat::R8Int,             "R8Int",             PixelFormatType::Sint,         1,     1,        { false, false },    { 8,  0,  0,  0  } },
        { PixelFormat::R8Uint,            "R8Uint",            PixelFormatType::Uint,         1,     1,        { false, false },    { 8,  0,  0,  0  } },
        { PixelFormat::R16Int,            "R16Int",            PixelFormatType::Sint,         2,     1,        { false, false },    { 16, 0,  0,  0  } },
        { PixelFormat::R16Uint,           "R16Uint",           PixelFormatType::Uint,         2,     1,        { false, false },    { 16, 0,  0,  0  } },
        { PixelFormat::R32Int,            "R32Int",            PixelFormatType::Sint,         4,     1,        { false, false },    { 32, 0,  0,  0  } },
        { PixelFormat::R32Uint,           "R32Uint",           PixelFormatType::Uint,         4,     1,        { false, false },    { 32, 0,  0,  0  } },
        { PixelFormat::RG8Int,            "RG8Int",            PixelFormatType::Sint,         2,     2,        { false, false },    { 8,  8,  0,  0  } },
        { PixelFormat::RG8Uint,           "RG8Uint",           PixelFormatType::Uint,         2,     2,        { false, false },    { 8,  8,  0,  0  } },
        { PixelFormat::RG16Int,           "RG16Int",           PixelFormatType::Sint,         4,     2,        { false, false },    { 16, 16, 0,  0  } },
        { PixelFormat::RG16Uint,          "RG16Uint",          PixelFormatType::Uint,         4,     2,        { false, false },    { 16, 16, 0,  0  } },
        { PixelFormat::RG32Int,           "RG32Int",           PixelFormatType::Sint,         8,     2,        { false, false },    { 32, 32, 0,  0  } },
        { PixelFormat::RG32Uint,          "RG32Uint",          PixelFormatType::Uint,         8,     2,        { false, false },    { 32, 32, 0,  0  } },
        { PixelFormat::RGB16Int,          "RGB16Int",          PixelFormatType::Sint,         6,     3,        { false, false },    { 16, 16, 16, 0  } },
        { PixelFormat::RGB16Uint,         "RGB16Uint",         PixelFormatType::Uint,         6,     3,        { false, false },    { 16, 16, 16, 0  } },
        { PixelFormat::RGB32Int,          "RGB32Int",          PixelFormatType::Sint,         12,    3,        { false, false },    { 32, 32, 32, 0  } },
        { PixelFormat::RGB32Uint,         "RGB32Uint",         PixelFormatType::Uint,         12,    3,        { false, false },    { 32, 32, 32, 0  } },
        { PixelFormat::RGBA8Int,          "RGBA8Int",          PixelFormatType::Sint,         4,     4,        { false, false },    { 8,  8,  8,  8  } },
        { PixelFormat::RGBA8Uint,         "RGBA8Uint",         PixelFormatType::Uint,         4,     4,        { false, false },    { 8,  8,  8,  8  } },
        { PixelFormat::RGBA16Int,         "RGBA16Int",         PixelFormatType::Sint,         8,     4,        { false, false },    { 16, 16, 16, 16 } },
        { PixelFormat::RGBA16Uint,        "RGBA16Uint",        PixelFormatType::Uint,         8,     4,        { false, false },    { 16, 16, 16, 16 } },
        { PixelFormat::RGBA32Int,         "RGBA32Int",         PixelFormatType::Sint,         16,    4,        { false, false },    { 32, 32, 32, 32 } },
        { PixelFormat::RGBA32Uint,        "RGBA32Uint",        PixelFormatType::Uint,         16,    4,        { false, false },    { 32, 32, 32, 32 } },
                                                                                                              
        { PixelFormat::BGRA8Unorm,        "BGRA8Unorm",        PixelFormatType::Unorm,        4,     4,        { false, false },    { 8,  8,  8,  8  } },
        { PixelFormat::BGRA8UnormSrgb,    "BGRA8UnormSrgb",    PixelFormatType::UnormSrgb,    4,     4,        { false, false },    { 8,  8,  8,  8  } },
                                                                                                              
        { PixelFormat::D32Float,          "D32Float",          PixelFormatType::Float,        4,     1,        { true,  false },    { 32, 0,  0,  0  } },
        { PixelFormat::D16Unorm,          "D16Unorm",          PixelFormatType::Unorm,        2,     1,        { true,  false },    { 16, 0,  0,  0  } },
        { PixelFormat::D24UnormS8Uint,    "D24UnormS8Uint",    PixelFormatType::Unorm,        4,     2,        { true,  true  },    { 24, 8,  0,  0  } },

        { PixelFormat::A2BGR10Unorm,      "A2BGR10Unorm",      PixelFormatType::Unorm,        4,     4,        { false, false },    { 2,  10, 10, 10 } },
    };
    static_assert(ARRAY_SIZE(GPixelFormatTable) == (uint32)PixelFormat::Count);

    void RenderBackendTick(RenderBackend* backend)
    {
	    backend->Tick(backend->instance);
    }

    void RenderBackendCreateRenderDevices(RenderBackend* backend, PhysicalDeviceID* physicalDeviceIDs, uint32 numDevices, uint32* outDeviceMasks)
    {
	    backend->CreateRenderDevices(backend->instance, physicalDeviceIDs, numDevices, outDeviceMasks);
    }

    void RenderBackendDestroyRenderDevices(RenderBackend* backend)
    {
	    backend->DestroyRenderDevices(backend->instance);
    }

	void RenderBackendFlushRenderDevices(RenderBackend* backend)
	{
		backend->FlushRenderDevices(backend->instance);
	}

    RenderBackendSwapChainHandle RenderBackendCreateSwapChain(RenderBackend* backend, uint32 deviceMask, uint64 windowHandle)
    {
	    return backend->CreateSwapChain(backend->instance, deviceMask, windowHandle);
    }

    void RenderBackendDestroySwapChain(RenderBackend* backend, RenderBackendSwapChainHandle swapChain)
    {
	    backend->DestroySwapChain(backend->instance, swapChain);
    }

    void RenderBackendResizeSwapChain(RenderBackend* backend, RenderBackendSwapChainHandle swapChain, uint32* width, uint32* height)
    {
	    backend->ResizeSwapChain(backend->instance, swapChain, width, height);
    }

    bool RenderBackendPresentSwapChain(RenderBackend* backend, RenderBackendSwapChainHandle swapChain)
    {
	    return backend->PresentSwapChain(backend->instance, swapChain);
    }

    RenderBackendTextureHandle RenderBackendGetActiveSwapChainBuffer(RenderBackend* backend, RenderBackendSwapChainHandle swapChain)
    {
	    return backend->GetActiveSwapChainBuffer(backend->instance, swapChain);
    }

    RenderBackendBufferHandle RenderBackendCreateBuffer(RenderBackend* backend, uint32 deviceMask, const RenderBackendBufferDesc* desc, const char* name)
    {
	    return backend->CreateBuffer(backend->instance, deviceMask, desc, name);
    }

    void RenderBackendResizeBuffer(RenderBackend* backend, RenderBackendBufferHandle buffer, uint64 size)
    {
	    backend->ResizeBuffer(backend->instance, buffer, size);
    }

	void RenderBackendMapBuffer(RenderBackend* backend, RenderBackendBufferHandle buffer, void** data)
	{
		backend->MapBuffer(backend->instance, buffer, data);
	}

	void RenderBackendUnmapBuffer(RenderBackend* backend, RenderBackendBufferHandle buffer)
	{
		backend->UnmapBuffer(backend->instance, buffer);
	}

    void RenderBackendDestroyBuffer(RenderBackend* backend, RenderBackendBufferHandle buffer)
    {
	    backend->DestroyBuffer(backend->instance, buffer);
    }

    RenderBackendTextureHandle RenderBackendCreateTexture(RenderBackend* backend, uint32 deviceMask, const RenderBackendTextureDesc* desc, const void* data, const char* name)
    {
	    return backend->CreateTexture(backend->instance, deviceMask, desc, data, name);
    }

    void RenderBackendDestroyTexture(RenderBackend* backend, RenderBackendTextureHandle texture)
    {
	    backend->DestroyTexture(backend->instance, texture);
    }

	void RenderBackendGetTextureReadbackData(RenderBackend* backend, RenderBackendTextureHandle texture, void** data)
	{
		backend->GetTextureReadbackData(backend->instance, texture, data);
	}

    RenderBackendTextureSRVHandle RenderBakendCreateTextureSRV(RenderBackend* backend, uint32 deviceMask, const RenderBackendTextureSRVDesc* desc, const char* name)
    {
	    return backend->CreateTextureSRV(backend->instance, deviceMask, desc, name);
    }

    int32 RenderBackendGetTextureSRVDescriptorIndex(RenderBackend* backend, uint32 deviceMask, RenderBackendTextureHandle srv)
    {
	    return backend->GetTextureSRVDescriptorIndex(backend->instance, deviceMask, srv);
    }

    RenderBackendTextureUAVHandle RenderBackendCreateTextureUAV(RenderBackend* backend, uint32 deviceMask, const RenderBackendTextureUAVDesc* desc, const char* name)
    {
	    return backend->CreateTextureUAV(backend->instance, deviceMask, desc, name);
    }

    int32 RenderBackendGetTextureUAVDescriptorIndex(RenderBackend* backend, uint32 deviceMask, RenderBackendTextureHandle uav)
    {
	    return backend->GetTextureUAVDescriptorIndex(backend->instance, deviceMask, uav);
    }

    RenderBackendSamplerHandle RenderBackendCreateSampler(RenderBackend* backend, uint32 deviceMask, const RenderBackendSamplerDesc* desc, const char* name)
    {
	    return backend->CreateSampler(backend->instance, deviceMask, desc, name);
    }

    void RenderBackendDestroySampler(RenderBackend* backend, RenderBackendSamplerHandle sampler)
    {
	    backend->DestroySampler(backend->instance, sampler);
    }

    RenderBackendShaderHandle RenderBackendCreateShader(RenderBackend* backend, uint32 deviceMask, const RenderBackendShaderDesc* desc, const char* name)
    {
	    return backend->CreateShader(backend->instance, deviceMask, desc, name);
    }

    void RenderBackendDestroyShader(RenderBackend* backend, RenderBackendShaderHandle shader)
    {
	    backend->DestroyShader(backend->instance, shader);
    }

	RenderBackendTimingQueryHeapHandle RenderBackendCreateTimingQueryHeap(RenderBackend* backend, uint32 deviceMask, const RenderBackendTimingQueryHeapDesc* desc, const char* name)
	{
		return backend->CreateTimingQueryHeap(backend->instance, deviceMask, desc, name);
	}

	void RenderBackendDestroyTimingQueryHeap(RenderBackend* backend, RenderBackendTimingQueryHeapHandle timingQueryHeap)
	{
		backend->DestroyTimingQueryHeap(backend->instance, timingQueryHeap);
	}

	bool RenderBackendGetTimingQueryHeapResults(RenderBackend* backend, RenderBackendTimingQueryHeapHandle timingQueryHeap, uint32 regionStart, uint32 regionCount, void* results)
	{
		return backend->GetTimingQueryHeapResults(backend->instance, timingQueryHeap, regionStart, regionCount, results);
	}

	RenderBackendOcclusionQueryHeapHandle RenderBackendCreateOcclusionQueryHeap(RenderBackend* backend, uint32 deviceMask, const RenderBackendOcclusionQueryHeapDesc* desc, const char* name)
	{
		return backend->CreateOcclusionQueryHeap(backend->instance, deviceMask, desc, name);
	}

	void RenderBackendDestroyOcclusionQueryHeap(RenderBackend* backend, RenderBackendOcclusionQueryHeapHandle occlusionQueryHeap)
	{
		backend->DestroyOcclusionQueryHeap(backend->instance, occlusionQueryHeap);
	}

    void RenderBackendSubmitRenderCommandLists(RenderBackend* backend, RenderCommandList** commandLists, uint32 numCommandLists)
    {
	    backend->SubmitRenderCommandLists(backend->instance, commandLists, numCommandLists);
    }

    void RenderBackendGetRenderStatistics(RenderBackend* backend, uint32 deviceMask, RenderStatistics* statistics)
    {
	    backend->GetRenderStatistics(backend->instance, deviceMask, statistics);
    }

    RenderBackendRayTracingAccelerationStructureHandle RenderBackendCreateBottomLevelAS(RenderBackend* backend, uint32 deviceMask, const RenderBackendBottomLevelASDesc* desc, const char* name)
    {
	    return backend->CreateBottomLevelAS(backend->instance, deviceMask, desc, name);
    }

    RenderBackendRayTracingAccelerationStructureHandle RenderBackendCreateTopLevelAS(RenderBackend* backend, uint32 deviceMask, const RenderBackendTopLevelASDesc* desc, const char* name)
    {
	    return backend->CreateTopLevelAS(backend->instance, deviceMask, desc, name);
    }

    RenderBackendRayTracingPipelineStateHandle RenderBackendCreateRayTracingPipelineState(RenderBackend* backend, uint32 deviceMask, const RenderBackendRayTracingPipelineStateDesc* desc, const char* name)
    {
	    return backend->CreateRayTracingPipelineState(backend->instance, deviceMask, desc, name);
    }

    RenderBackendBufferHandle RenderBackendCreateRayTracingShaderBindingTable(RenderBackend* backend, uint32 deviceMask, const RenderBackendRayTracingShaderBindingTableDesc* desc, const char* name)
    {
	    return backend->CreateRayTracingShaderBindingTable(backend->instance, deviceMask, desc, name);
    }

	void RenderCommandList::CopyTexture2D(RenderBackendTextureHandle srcTexture, const Offset2D& srcOffset, uint32 srcMipLevel, RenderBackendTextureHandle dstTexture, const Offset2D& dstOffset, uint32 dstMipLevel, const Extent2D extent)
	{
		RenderCommandCopyTexture* command = AllocateCommand<RenderCommandCopyTexture>(RenderCommandCopyTexture::Type);
		command->srcTexture = srcTexture;
		command->srcOffset = {
			.x = srcOffset.x,
			.y = srcOffset.y,
			.z = 0,
		};
		command->srcSubresourceLayers = {
			.mipLevel = srcMipLevel,
			.firstLayer = 0,
			.arrayLayers = 1,
		};
		command->dstTexture = dstTexture;
		command->dstOffset = {
			.x = dstOffset.x,
			.y = dstOffset.y,
			.z = 0,
		};
		command->dstSubresourceLayers = {
			.mipLevel = dstMipLevel,
			.firstLayer = 0,
			.arrayLayers = 1,
		};
		command->extent = {
			.width = extent.width,
			.height = extent.height,
			.depth = 1,
		};
	}

	void RenderCommandList::CopyBuffer(RenderBackendBufferHandle srcBuffer, uint64 srcOffset, RenderBackendBufferHandle dstBuffer, uint64 dstOffset, uint64 bytes)
	{
		RenderCommandCopyBuffer* command = AllocateCommand<RenderCommandCopyBuffer>(RenderCommandCopyBuffer::Type);
		command->srcBuffer = srcBuffer;
		command->srcOffset = srcOffset;
		command->dstBuffer = dstBuffer;
		command->dstOffset = dstOffset;
		command->bytes = bytes;
	}

	void RenderCommandList::UpdateBuffer(RenderBackendBufferHandle buffer, uint64 offset, const void* data, uint64 size)
	{
		RenderCommandUpdateBuffer* command = AllocateCommand<RenderCommandUpdateBuffer>(RenderCommandUpdateBuffer::Type);
		command->buffer = buffer;
		command->offset = offset;
		command->data = data;
		command->size = size;
	}

	void RenderCommandList::Dispatch(RenderBackendShaderHandle shader, const ShaderArguments& shaderArguments, uint32 x, uint32 y, uint32 z)
	{
		RenderCommandDispatch* command = AllocateCommand<RenderCommandDispatch>(RenderCommandDispatch::Type);
		command->shader = shader;
		command->threadGroupCountX = x;
		command->threadGroupCountY = y;
		command->threadGroupCountZ = z;
		memcpy(&command->shaderArguments, &shaderArguments, sizeof(ShaderArguments));
	}

	void RenderCommandList::Dispatch2D(RenderBackendShaderHandle shader, const ShaderArguments& shaderArguments, uint32 x, uint32 y)
	{
		Dispatch(shader, shaderArguments, x, y, 1);
	}

	void RenderCommandList::TraceRays(RenderBackendRayTracingPipelineStateHandle pipelineState, RenderBackendBufferHandle shaderBindingTable, const ShaderArguments& shaderArguments, uint32 x, uint32 y, uint32 z)
	{
		RenderCommandTraceRays* command = AllocateCommand<RenderCommandTraceRays>(RenderCommandTraceRays::Type);
		command->pipelineState = pipelineState;
		command->shaderBindingTable = shaderBindingTable;
		command->width = x;
		command->height = y;
		command->depth = z;
		memcpy(&command->shaderArguments, &shaderArguments, sizeof(ShaderArguments));
	}

	void RenderCommandList::SetViewports(RenderBackendViewport* viewports, uint32 numViewports)
	{
		RenderCommandSetViewport* command = AllocateCommand<RenderCommandSetViewport>(RenderCommandSetViewport::Type);
		command->numViewports = numViewports;
		memcpy(command->viewports, viewports, numViewports * sizeof(RenderBackendViewport));
	}

	void RenderCommandList::SetScissors(RenderBackendScissor* scissors, uint32 numScissors)
	{
		RenderCommandSetScissor* command = AllocateCommand<RenderCommandSetScissor>(RenderCommandSetScissor::Type);
		command->numScissors = numScissors;
		memcpy(command->scissors, scissors, numScissors * sizeof(RenderBackendScissor));
	}

	void RenderCommandList::BeginRenderPass(const RenderPassInfo& renderPassInfo)
	{
		RenderCommandBeginRenderPass* command = AllocateCommand<RenderCommandBeginRenderPass>(RenderCommandBeginRenderPass::Type);
		memcpy(command, &renderPassInfo, sizeof(RenderPassInfo));
	}

	void RenderCommandList::EndRenderPass()
	{
		RenderCommandEndRenderPass* command = AllocateCommand<RenderCommandEndRenderPass>(RenderCommandEndRenderPass::Type);
	}

	void RenderCommandList::Transitions(RenderBackendBarrier* transitions, uint32 numTransitions)
	{
		RenderCommandTransitions* command = AllocateCommand<RenderCommandTransitions>(RenderCommandTransitions::Type, sizeof(RenderCommandTransitions) + numTransitions * sizeof(RenderBackendBarrier));
		command->numTransitions = numTransitions;
		command->transitions = (RenderBackendBarrier*)(((uint8*)command) + sizeof(RenderCommandTransitions));
		memcpy(command->transitions, transitions, numTransitions * sizeof(RenderBackendBarrier));
	}

	void RenderCommandList::Draw(RenderBackendShaderHandle shader, const ShaderArguments& shaderArguments, uint32 numVertices, uint32 numInstances, uint32 firstVertex, uint32 firstInstance, PrimitiveTopology topology)
	{
		RenderCommandDraw* command = AllocateCommand<RenderCommandDraw>(RenderCommandDraw::Type);
		command->shader = shader;
		command->numVertices = numVertices;
		command->numInstances = numInstances;
		command->firstVertex = firstVertex;
		command->firstInstance = firstInstance;
		command->topology = topology;
		command->indexBuffer = RenderBackendBufferHandle::NullHandle;
		memcpy(&command->shaderArguments, &shaderArguments, sizeof(ShaderArguments));
	}

	void RenderCommandList::DrawIndexed(
		RenderBackendShaderHandle shader,
		const ShaderArguments& shaderArguments,
		RenderBackendBufferHandle indexBuffer,
		uint32 numIndices,
		uint32 numInstances,
		uint32 firstIndex,
		int32 vertexOffset,
		uint32 firstInstance,
		PrimitiveTopology topology)
	{
		RenderCommandDraw* command = AllocateCommand<RenderCommandDraw>(RenderCommandDraw::Type);
		command->shader = shader;
		command->indexBuffer = indexBuffer;
		command->numIndices = numIndices;
		command->numInstances = numInstances;
		command->firstIndex = firstIndex;
		command->vertexOffset = vertexOffset;
		command->firstInstance = firstInstance;
		command->topology = topology;
		memcpy(&command->shaderArguments, &shaderArguments, sizeof(ShaderArguments));
	}

	void RenderCommandList::DrawIndirect(
		RenderBackendShaderHandle shader,
		const ShaderArguments& shaderArguments,
		RenderBackendBufferHandle indexBuffer,
		RenderBackendBufferHandle argumentBuffer,
		uint64 offset,
		uint32 numDraws,
		uint32 stride,
		PrimitiveTopology topology)
	{
		RenderCommandDrawIndirect* command = AllocateCommand<RenderCommandDrawIndirect>(RenderCommandDrawIndirect::Type);
		command->shader = shader;
		command->indexBuffer = indexBuffer;
		command->argumentBuffer = argumentBuffer;
		command->offset = offset;
		command->numDraws = numDraws;
		command->stride = stride;
		command->topology = topology;
		memcpy(&command->shaderArguments, &shaderArguments, sizeof(ShaderArguments));
	}

	void RenderCommandList::DrawIndexedIndirect(
		RenderBackendShaderHandle shader,
		const ShaderArguments& shaderArguments,
		RenderBackendBufferHandle indexBuffer,
		RenderBackendBufferHandle argumentBuffer,
		uint64 offset,
		uint32 numDraws,
		uint32 stride,
		PrimitiveTopology topology)
	{
		RenderCommandDrawIndirect* command = AllocateCommand<RenderCommandDrawIndirect>(RenderCommandDrawIndirect::Type);
		command->shader = shader;
		command->indexBuffer = indexBuffer;
		command->argumentBuffer = argumentBuffer;
		command->offset = offset;
		command->numDraws = numDraws;
		command->stride = stride;
		command->topology = topology;
		memcpy(&command->shaderArguments, &shaderArguments, sizeof(ShaderArguments));
	}

	void RenderCommandList::BeginTimingQuery(RenderBackendTimingQueryHeapHandle timingQueryHeap, uint32 region)
	{
		auto* command = AllocateCommand<RenderCommandBeginTimingQuery>(RenderCommandBeginTimingQuery::Type);
		command->timingQueryHeap = timingQueryHeap;
		command->region = region;
	}

	void RenderCommandList::EndTimingQuery(RenderBackendTimingQueryHeapHandle timingQueryHeap, uint32 region)
	{
		auto* command = AllocateCommand<RenderCommandEndTimingQuery>(RenderCommandEndTimingQuery::Type);
		command->timingQueryHeap = timingQueryHeap;
		command->region = region;
	}

	void RenderCommandList::ResetTimingQueryHeap(RenderBackendTimingQueryHeapHandle timingQueryHeap, uint32 regionStart, uint32 regionCount)
	{
		auto* command = AllocateCommand<RenderCommandResetTimingQueryHeap>(RenderCommandResetTimingQueryHeap::Type);
		command->timingQueryHeap = timingQueryHeap;
		command->regionStart = regionStart;
		command->regionCount = regionCount;
	}

	void LoadShaderSourceFromFile(const char* filename, std::vector<uint8>& outData)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);
		if (!file.is_open())
		{
			// HE_LOG_ERROR("Failed to open shader source file.");
			return;
		}
		size_t fileSize = (size_t)file.tellg();
		outData.resize(fileSize);
		file.seekg(0);
		file.read((char*)outData.data(), fileSize);
		file.close();
	}

	bool CompileShader(
		ShaderCompiler* compiler,
		std::vector<uint8> source,
		const wchar* entry,
		RenderBackendShaderStage stage,
		ShaderRepresentation representation,
		const std::vector<const wchar*>& includeDirs,
		const std::vector<const wchar*>& defines,
		ShaderBlob* outBlob)
	{
		return compiler->CompileShader(compiler->instance, source, entry, stage, representation, includeDirs, defines, outBlob);
	}

	void ReleaseShaderBlob(ShaderCompiler* compiler, ShaderBlob* blob)
	{
		return compiler->ReleaseShaderBlob(compiler->instance, blob);
	}
}