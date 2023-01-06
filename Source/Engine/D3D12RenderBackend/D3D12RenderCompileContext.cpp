#include "D3D12RenderCompileContext.h"

namespace HE
{
	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandCopyBuffer& command)
	{
		const auto& srcBuffer = device->GetBuffer(command.srcBuffer);
		const auto& dstBuffer = device->GetBuffer(command.dstBuffer);
		commandList->CopyBufferRegion(dstBuffer, command.dstOffset, srcBuffer, command.srcOffset, command.bytes);
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandCopyTexture& command)
	{
		const auto& srcTexture = device->GetTexture(command.srcTexture);
		const auto& dstTexture = device->GetTexture(command.dstTexture);
		UINT srcSubresource = D3D12CalcSubresource(command.srcSubresourceLayers.mipLevel, command.srcSubresourceLayers.firstLayer, 0, 1, command.srcSubresourceLayers.arrayLayers);
		UINT dstSubresource = D3D12CalcSubresource(command.dstSubresourceLayers.mipLevel, command.dstSubresourceLayers.firstLayer, 0, 1, command.dstSubresourceLayers.arrayLayers);
		CD3DX12_TEXTURE_COPY_LOCATION srcCopyLocation(srcTexture, srcSubresource);
		CD3DX12_TEXTURE_COPY_LOCATION dstCopyLocation(dstTexture, dstSubresource);
		D3D12_BOX srcRegion = {
			.left = command.srcOffset.x,
			.top = command.srcOffset.y,
			.front = command.srcOffset.z,
			.right = command.srcOffset.x + command.extent.width,
			.bottom = command.srcOffset.y + command.extent.height,
			.back = command.srcOffset.z + command.extent.depth,
		};
		commandList->CopyTextureRegion(&dstCopyLocation, command.dstOffset.x, command.dstOffset.y, command.dstOffset.z, &srcCopyLocation, &srcRegion);
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandBarriers& command)
	{
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandTransitions& command)
	{
		for (uint32 i = 0; i < command.numTransitions; i++)
		{
			const auto& transition = command.transitions[i];
			ASSERT(transition.srcState != transition.dstState);
			if (transition.type == RenderBackendBarrier::ResourceType::Texture)
			{
				VulkanTexture* texture = device->GetTexture(transition.texture);
				VkPipelineStageFlags2 srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				VkPipelineStageFlags2 dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
				VkAccessFlags2 srcAccessMask, dstAccessMask;
				VkImageLayout oldLayout, newLayout;
				GetBarrierInfo2(
					transition.srcState,
					transition.dstState,
					&oldLayout,
					&newLayout,
					&srcStageMask,
					&dstStageMask,
					&srcAccessMask,
					&dstAccessMask);
				VkImageMemoryBarrier2 imageBarrier = {
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
					.srcStageMask = srcStageMask,
					.srcAccessMask = srcAccessMask,
					.dstStageMask = dstStageMask,
					.dstAccessMask = dstAccessMask,
					.oldLayout = oldLayout,
					.newLayout = newLayout,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = texture->handle,
					.subresourceRange = {
						.aspectMask = texture->aspectMask,
						.baseMipLevel = transition.textureRange.firstLevel,
						.levelCount = transition.textureRange.mipLevels,
						.baseArrayLayer = transition.textureRange.firstLayer,
						.layerCount = transition.textureRange.arrayLayers,
					},
				};
				imageBarriers.push_back(std::move(imageBarrier));
			}
			else if (transition.type == RenderBackendBarrier::ResourceType::Buffer)
			{
				/*VulkanBuffer* buffer = device->GetBuffer(transition.buffer);
				VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
				VkAccessFlags srcAccessMask, dstAccessMask;
				GetBarrierInfo(
					transition.srcState,
					transition.dstState,
					nullptr,
					nullptr,
					&srcStageMask,
					&dstStageMask,
					&srcAccessMask,
					&dstAccessMask
				);
				VkBufferMemoryBarrier bufferMemoryBarrier = {
					.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
					.srcAccessMask = srcAccessMask,
					.dstAccessMask = dstAccessMask,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.buffer = buffer->handle,
					.offset = transition.bufferRange.offset,
					.size = transition.bufferRange.size,
				};
				vkCmdPipelineBarrier(
					commandBuffer,
					srcStageMask,
					dstStageMask,
					0,
					0,
					nullptr,
					1,
					&bufferMemoryBarrier,
					0,
					nullptr);*/
			}
		}
		VkDependencyInfo dependency = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			//.bufferMemoryBarrierCount = ,
			//.pBufferMemoryBarriers = ,
			.imageMemoryBarrierCount = (uint32)imageBarriers.size(),
			.pImageMemoryBarriers = imageBarriers.data(),
		};
		vkCmdPipelineBarrier2(commandBuffer, &dependency);
		imageBarriers.clear();
		statistics.transitions += command.numTransitions;
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandBeginTimingQuery& command)
	{
		const auto& timingQueryHeap = device->GetTimingQueryHeap(command.timingQueryHeap);
		uint32 queryIndex = command.region * 2 + 0;
		ASSERT(queryIndex < timingQueryHeap->maxQueryCount);
		commandList->BeginQuery(timingQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, queryIndex);
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandEndTimingQuery& command)
	{
		const auto& timingQueryHeap = device->GetTimingQueryHeap(command.timingQueryHeap);
		uint32 queryIndex = command.region * 2 + 1;
		ASSERT(queryIndex < timingQueryHeap->maxQueryCount);
		commandList->EndQuery(timingQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, queryIndex);
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandResolveTimings& command)
	{
		//const auto& timingQueryPool = device->GetTimingQueryPool(command.timingQueryPool);
		//const auto& readbackBuffer = timingQueryPool->readbackBuffer;

		//uint32 firstQuery = 2 * command.regionStart;
		//uint32 queryCount = 2 * command.regionCount;

		//vkCmdCopyQueryPoolResults(
		//	commandBuffer,
		//	timingQueryPool->GetHandle(),
		//	firstQuery,
		//	queryCount,
		//	readbackBuffer->GetHandle(),
		//	sizeof(uint64) * firstQuery,
		//	sizeof(uint64),
		//	VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
		//);

		//vkCmdResetQueryPool(commandBuffer, timingQueryPool->GetHandle(), firstQuery, queryCount);
		return true;
	}

	void D3D12RenderCompileContext::ApplyTransitions()
	{
		if (!bufferBarriers.empty() || !imageBarriers.empty())
		{
			VkDependencyInfoKHR dependencyInfo = {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
				.bufferMemoryBarrierCount = (uint32)bufferBarriers.size(),
				.pBufferMemoryBarriers = bufferBarriers.data(),
				.imageMemoryBarrierCount = (uint32)imageBarriers.size(),
				.pImageMemoryBarriers = imageBarriers.data(),
			};
			device->GetBackend()->functions.vkCmdPipelineBarrier2KHR(commandBuffer, &dependencyInfo);
			bufferBarriers.clear();
			imageBarriers.clear();
		}
	}

	bool D3D12RenderCompileContext::PrepareForDispatch(RenderBackendShaderHandle shader, const ShaderArgument& shaderArgument)
	{
		VulkanPushConstants pushConstants;
		for (uint32 i = 0; i < 16; i++)
		{
			if (shaderArgument.slots[i].type == ShaderArgument::Slot::Type::Buffer)
			{
				VulkanBuffer* buffer = device->GetBuffer(shaderArgument.slots[i].buffer.handle);
				pushConstants.indices[i] = (buffer->uavIndex << 16) | (uint16)shaderArgument.slots[i].buffer.offset;
			}
			else if (shaderArgument.slots[i].type == ShaderArgument::Slot::Type::Texture)
			{
				VulkanTexture* texture = device->GetTexture(shaderArgument.slots[i].texture.handle);
				pushConstants.indices[i] = shaderArgument.slots[i].texture.srvOrUav ? texture->srvIndex : texture->uavIndex;
			}
		}
		const void* pushConstantsValue = &pushConstants;
		uint32 pushConstantsSize = sizeof(VulkanPushConstants);

		VulkanPipeline* pipeline = device->FindOrCreateComputePipeline(device->GetShader(shader), pushConstantsSize);
		if (pipeline->handle != activeComputePipeline)
		{
			VkDescriptorSet set = device->GetBindlessGlobalSet();
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &set, 0, nullptr);
			activeComputePipeline = pipeline->handle;
			statistics.pipelines++;
		}
		if (pushConstantsSize)
		{
			vkCmdPushConstants(commandBuffer, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantsSize, pushConstantsValue);
		}
		ApplyTransitions();
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandDispatch& command)
	{
		if (!PrepareForDispatch(command.shader, command.shaderArgument))
		{
			return false;
		}
		vkCmdDispatch(commandBuffer, command.threadGroupCountX, command.threadGroupCountY, command.threadGroupCountZ);
		statistics.computeDispatches++;
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandDispatchIndirect& command)
	{
		if (!PrepareForDispatch(command.shader, command.shaderArgument))
		{
			return false;
		}
		vkCmdDispatchIndirect(commandBuffer, device->GetBuffer(command.argumentBuffer)->handle, command.argumentOffset);
		statistics.computeIndirectDispatches++;
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandUpdateBottomLevelAS& command)
	{
		const auto& srcBLAS = command.srcBLAS ? device->GetAccelerationStructure(command.srcBLAS) : nullptr;
		const auto& dstBLAS = device->GetAccelerationStructure(command.dstBLAS);

		VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = {
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
			.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
			.flags = dstBLAS->buildFlags,
			.mode = srcBLAS ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
			.srcAccelerationStructure = srcBLAS ? srcBLAS->handle : VK_NULL_HANDLE,
			.dstAccelerationStructure = dstBLAS->handle,
			.geometryCount = (uint32)dstBLAS->geometries.size(),
			.pGeometries = dstBLAS->geometries.data(),
			.scratchData = dstBLAS->scratchBuffer.deviceAddress
		};

		uint32 numGeometries = (uint32)dstBLAS->blasDesc.numGeometries;
		std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(numGeometries);
		std::vector<VkAccelerationStructureBuildRangeInfoKHR*> buildRangeInfos(numGeometries);
		for (uint32 i = 0; i < numGeometries; i++)
		{
			if (dstBLAS->blasDesc.geometryDescs->type == RenderBackendGeometryType::Triangles)
			{
				VkAccelerationStructureBuildRangeInfoKHR buildRange = {
					.primitiveCount = dstBLAS->blasDesc.geometryDescs[i].triangleDesc.numIndices / 3,
					.primitiveOffset = 0,
					.firstVertex = 0,
					.transformOffset = dstBLAS->blasDesc.geometryDescs[i].triangleDesc.transformOffset,
				};
				ranges[i] = buildRange;
				buildRangeInfos[i] = &ranges[i];
			}
			else if (dstBLAS->blasDesc.geometryDescs->type == RenderBackendGeometryType::AABBs)
			{
				VulkanBuffer* buffer = device->GetBuffer(dstBLAS->blasDesc.geometryDescs[i].aabbDesc.buffer);
				VkAccelerationStructureBuildRangeInfoKHR buildRange = {
					.primitiveCount = (uint32)(buffer->size / sizeof(VkAabbPositionsKHR)),
					.primitiveOffset = dstBLAS->blasDesc.geometryDescs[i].aabbDesc.offset,
					.firstVertex = 0,
					.transformOffset = 0,
				};
				ranges[i] = buildRange;
				buildRangeInfos[i] = &ranges[i];
			}
		}

		device->GetBackend()->functions.vkCmdBuildAccelerationStructuresKHR(
			commandBuffer,
			1,
			&accelerationStructureBuildGeometryInfo,
			buildRangeInfos.data());
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandUpdateTopLevelAS& command)
	{
		const auto& srcTLAS = command.srcTLAS ? device->GetAccelerationStructure(command.srcTLAS) : nullptr;
		const auto& dstTLAS = device->GetAccelerationStructure(command.dstTLAS);

		VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = {
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
			.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
			.flags = dstTLAS->buildFlags,
			.mode = srcTLAS ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
			.srcAccelerationStructure = srcTLAS ? srcTLAS->handle : VK_NULL_HANDLE,
			.dstAccelerationStructure = dstTLAS->handle,
			.geometryCount = (uint32)dstTLAS->geometries.size(),
			.pGeometries = dstTLAS->geometries.data(),
			.scratchData = dstTLAS->scratchBuffer.deviceAddress,
		};
		ASSERT((uint32)dstTLAS->geometries.size() == 1);

		VkAccelerationStructureBuildRangeInfoKHR range = {
			.primitiveCount = dstTLAS->tlasDesc.numInstances,
			.primitiveOffset = 0,
			.firstVertex = 0,
			.transformOffset = 0,
		};
		VkAccelerationStructureBuildRangeInfoKHR* buildRangeInfo = &range;

		device->GetBackend()->functions.vkCmdBuildAccelerationStructuresKHR(
			commandBuffer,
			1,
			&accelerationStructureBuildGeometryInfo,
			&buildRangeInfo);

		// TODO: Destroy Buffers

		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandTraceRays& command)
	{
		if (!command.rgenSBT || !command.rmissSBT || command.rchitSBT)
		{
			return false;
		}

		const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rayTracingPipelineProperties = device->GetRayTracingPipelineProperties();
		const uint32 handleSizeAligned = GetHandleSizeAligned(rayTracingPipelineProperties.shaderGroupHandleSize, rayTracingPipelineProperties.shaderGroupHandleAlignment);

		VkStridedDeviceAddressRegionKHR raygenShaderBindingTable = {
			.deviceAddress = device->GetBufferDeviceAddress(command.rgenSBT),
			.stride = handleSizeAligned,
			.size = handleSizeAligned
		};
		VkStridedDeviceAddressRegionKHR missShaderBindingTable = {
			.deviceAddress = device->GetBufferDeviceAddress(command.rmissSBT),
			.stride = handleSizeAligned,
			.size = handleSizeAligned
		};
		VkStridedDeviceAddressRegionKHR hitShaderBindingTable = {
			.deviceAddress = device->GetBufferDeviceAddress(command.rchitSBT),
			.stride = handleSizeAligned,
			.size = handleSizeAligned
		};
		VkStridedDeviceAddressRegionKHR callableShaderBindingTable = {};

		uint32 pushConstantsSize = sizeof(VulkanPushConstants);
		const void* pushConstantsValue = &command.shaderArgument;
		VulkanPipeline* pipeline = device->FindOrCreateRayTracingPipeline(device->GetShader(command.shader), pushConstantsSize);
		if (pipeline->handle != activeComputePipeline)
		{
			VkDescriptorSet set = device->GetBindlessGlobalSet();
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline->layout, 0, 1, &set, 0, nullptr);
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline->handle);
			activeComputePipeline = pipeline->handle;
			statistics.pipelines++;
		}
		if (pushConstantsSize)
		{
			vkCmdPushConstants(commandBuffer, pipeline->layout, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, pushConstantsSize, pushConstantsValue);
		}

		device->GetBackend()->functions.vkCmdTraceRaysKHR(
			commandBuffer,
			&raygenShaderBindingTable,
			&missShaderBindingTable,
			&hitShaderBindingTable,
			&callableShaderBindingTable,
			command.width,
			command.height,
			command.depth);
		statistics.traceRayDispatches++;
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandBeginRenderPass& command)
	{
		VkClearValue clearValues[MaxNumSimultaneousColorRenderTargets + 1] = {};

		VulkanRenderPassDesc renderPassDesc = {};
		GetRenderPassDescAndClearValues(device, command.renderPassInfo, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, &renderPassDesc, clearValues);

		const auto& renderPass = device->FindOrCreateRenderPass(renderPassDesc);
		const auto& framebuffer = device->FindOrCreateFramebuffer(command.renderPassInfo, renderPassDesc, renderPass);
		if (!renderPass || !framebuffer)
		{
			return false;
		}

		VkRenderPassBeginInfo beginInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = renderPass,
			.framebuffer = framebuffer->handle,
			.renderArea = { 0, 0, framebuffer->width, framebuffer->height },
			.clearValueCount = framebuffer->numAttachments,
			.pClearValues = clearValues,
		};
		//
		// 
		clearValues[0].color.float32[0] = 0.5f;
		clearValues[0].color.float32[1] = 0.5f;
		clearValues[0].color.float32[2] = 1.0f;
		clearValues[0].color.float32[3] = 1.0f;
		clearValues[1].depthStencil.depth = 1.0f;
		//
		vkCmdBeginRenderPass(commandBuffer, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);
		activeRenderPass = renderPass;
		statistics.renderPasses++;
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandEndRenderPass& command)
	{
		vkCmdEndRenderPass(commandBuffer);
		activeRenderPass = VK_NULL_HANDLE;
		return true;
	}

	bool D3D12RenderCompileContext::PrepareForDraw(RenderBackendShaderHandle shader, PrimitiveTopology topology, RenderBackendBufferHandle indexBuffer, const ShaderArgument& shaderArgument)
	{
		ASSERT(activeRenderPass != VK_NULL_HANDLE);
		VulkanPushConstants pushConstants;
		for (uint32 i = 0; i < 16; i++)
		{
			if (shaderArgument.slots[i].type == ShaderArgument::Slot::Type::Buffer)
			{
				VulkanBuffer* buffer = device->GetBuffer(shaderArgument.slots[i].buffer.handle);
				pushConstants.indices[i] = (buffer->uavIndex << 16) | (shaderArgument.slots[i].buffer.offset & 0xffff);
			}
			else if (shaderArgument.slots[i].type == ShaderArgument::Slot::Type::Texture)
			{
				VulkanTexture* texture = device->GetTexture(shaderArgument.slots[i].texture.handle);
				pushConstants.indices[i] = shaderArgument.slots[i].texture.srvOrUav ? texture->srvIndex : texture->uavIndex;
			}
			pushConstants.data[i] = shaderArgument.data[i];
		}
		const void* pushConstantsValue = &pushConstants;
		uint32 pushConstantsSize = sizeof(VulkanPushConstants);
		VulkanPipeline* pipeline = device->FindOrCreateGraphicsPipeline(device->GetShader(shader), activeRenderPass, topology, pushConstantsSize);
		if (pipeline->handle != activeGraphicsPipeline)
		{
			VkDescriptorSet set = device->GetBindlessGlobalSet();
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->handle);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->layout, 0, 1, &set, 0, nullptr);
			activeGraphicsPipeline = pipeline->handle;
			statistics.pipelines++;
		}
		if (pushConstantsSize)
		{
			vkCmdPushConstants(commandBuffer, pipeline->layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, pushConstantsSize, pushConstantsValue);
		}
		if (indexBuffer)
		{
			vkCmdBindIndexBuffer(commandBuffer, device->GetBuffer(indexBuffer)->handle, 0, VK_INDEX_TYPE_UINT32);
		}
		ApplyTransitions();
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandDraw& command)
	{
		if (!PrepareForDraw(command.shader, command.topology, command.indexBuffer, command.shaderArgument))
		{
			return false;
		}
		if (!command.indexBuffer)
		{
			vkCmdDraw(
				commandBuffer,
				command.numVertices,
				command.numInstances,
				command.firstVertex,
				command.firstInstance
			);
			statistics.nonIndexedDraws++;
			statistics.vertices += command.numVertices * command.numInstances;
		}
		else
		{
			vkCmdDrawIndexed(
				commandBuffer,
				command.numIndices,
				command.numInstances,
				command.firstIndex,
				command.vertexOffset,
				command.firstInstance
			);
			statistics.indexedDraws++;
			statistics.vertices += command.numVertices * command.numInstances;
		}
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandDrawIndirect& command)
	{
		if (!PrepareForDraw(command.shader, command.topology, command.indexBuffer, command.shaderArgument))
		{
			return false;
		}
		if (!command.indexBuffer)
		{
			vkCmdDrawIndirect(
				commandBuffer,
				device->GetBuffer(command.argumentBuffer)->handle,
				command.offset,
				command.numDraws,
				command.stride
			);
			statistics.nonIndexedIndirectDraws++;
		}
		else
		{
			vkCmdDrawIndexedIndirect(
				commandBuffer,
				device->GetBuffer(command.argumentBuffer)->handle,
				command.offset,
				command.numDraws,
				command.stride
			);
			statistics.indexedIndirectDraws++;
		}
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandSetViewport& command)
	{
		VkViewport viewports[MaxNumViewports];
		for (uint32 i = 0; i < command.numViewports; i++)
		{
			viewports[i] = {
				.x = command.viewports[i].x,
				.y = command.viewports[i].y + command.viewports[i].height,
				.width = command.viewports[i].width,
				.height = -command.viewports[i].height,
				.minDepth = command.viewports[i].minDepth,
				.maxDepth = command.viewports[i].maxDepth
			};
		}
		vkCmdSetViewport(commandBuffer, 0, command.numViewports, viewports);
		return true;
	}

	bool D3D12RenderCompileContext::CompileRenderCommand(const RenderCommandSetScissor& command)
	{
		VkRect2D scissors[MaxNumViewports];
		for (uint32 i = 0; i < command.numScissors; i++)
		{
			scissors[i] = {
				.offset = {.x = command.scissors[i].left, .y = command.scissors[i].top },
				.extent = {.width = command.scissors[i].width, .height = command.scissors[i].height }
			};
		}
		vkCmdSetScissor(commandBuffer, 0, command.numScissors, scissors);
		return true;
	}

	struct BuildCommandBufferJobData
	{
		VulkanRenderBackend* backend;
		VulkanDevice* device;
		QueueFamily queueFamily;
		VkCommandBuffer commandBuffer;
		RenderCommandContainer* commandContainer;
		RenderStatistics statistics;
	};

	static void BuildCommandBuffer(BuildCommandBufferJobData* data)
	{
		VulkanRenderCompileContext context(data->device, data->queueFamily, data->commandBuffer);
		if (context.CompileRenderCommands(*data->commandContainer))
		{
			data->statistics = context.GetRenderStatistics();
		}
		else
		{
			// TODO
		}
	}

#define COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandStructType)                                         \
bool Compile##RenderCommandStructType(D3D12RenderCompileContext* context, void* command)                \
{	                                                                                                     \
	return context->CompileRenderCommand(*reinterpret_cast<const RenderCommandStructType*>(command));    \
}

	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandCopyBuffer);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandCopyTexture);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandBarriers);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandTransitions);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandBeginTimingQuery);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandEndTimingQuery);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandResolveTimings);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandDispatch);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandDispatchIndirect);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandUpdateBottomLevelAS);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandUpdateTopLevelAS);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandTraceRays);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandSetViewport);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandSetScissor);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandBeginRenderPass);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandEndRenderPass);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandDraw);
	COMPILE_RENDER_COMMAND_FUNCTION(RenderCommandDrawIndirect);

	/**
	 * Jump table of compile command functions.
	 */
	bool (*gCompileRenderCommandFunctions[])(D3D12RenderCompileContext*, void*) = {
		CompileRenderCommandCopyBuffer,
		CompileRenderCommandCopyTexture,
		CompileRenderCommandBarriers,
		CompileRenderCommandTransitions,
		CompileRenderCommandBeginTimingQuery,
		CompileRenderCommandEndTimingQuery,
		CompileRenderCommandResolveTimings,
		CompileRenderCommandDispatch,
		CompileRenderCommandDispatchIndirect,
		CompileRenderCommandUpdateBottomLevelAS,
		CompileRenderCommandUpdateTopLevelAS,
		CompileRenderCommandTraceRay,
		CompileRenderCommandSetViewport,
		CompileRenderCommandSetScissor,
		CompileRenderCommandBeginRenderPass,
		CompileRenderCommandEndRenderPass,
		CompileRenderCommandDraw,
		CompileRenderCommandDrawIndirect,
	};
	static_assert(ARRAY_SIZE(gCompileRenderCommandFunctions) == (int)RenderCommandType::Count);

	bool D3D12RenderCompileContext::CompileRenderCommands(const RenderCommandContainer& container)
	{
#define COMPILE_RENDER_COMMAND(type, command) gCompileRenderCommandFunctions[(int)type](this, command)
		for (uint32 i = 0; i < container.numCommands; i++)
		{
			if (!COMPILE_RENDER_COMMAND(container.types[i], container.commands[i]))
			{
				return false;
			}
		}
#undef COMPILE_RENDER_COMMAND_MACRO
		return true;
	}
}