#pragma once

#include "D3D12RenderBackend/D3D12Common.h"

namespace HE
{
	class D3D12RenderCompileContext
	{
	public:
		D3D12RenderCompileContext(ID3D12Device* device, QueueFamily family, VkCommandBuffer commandBuffer)
			: device(device)
			, queueFamily(family)
			, commandBuffer(commandBuffer)
			, activeRenderPass(VK_NULL_HANDLE)
			, activeComputePipeline(VK_NULL_HANDLE)
			, activeGraphicsPipeline(VK_NULL_HANDLE)
			, statistics()
			, imageBarriers()
			, bufferBarriers() {}
		virtual ~D3D12RenderCompileContext() = default;
		inline QueueFamily GetQueueFamily() const { return queueFamily; }
		inline VkCommandBuffer GetCommandBuffer() const { return commandBuffer; }
		inline RenderStatistics GetRenderStatistics() const { return statistics; }
		bool CompileRenderCommands(const RenderCommandContainer& container);
		bool CompileRenderCommand(const RenderCommandCopyBuffer& command);
		bool CompileRenderCommand(const RenderCommandCopyTexture& command);
		bool CompileRenderCommand(const RenderCommandBarriers& command);
		bool CompileRenderCommand(const RenderCommandTransitions& command);
		bool CompileRenderCommand(const RenderCommandBeginTimingQuery& command);
		bool CompileRenderCommand(const RenderCommandEndTimingQuery& command);
		bool CompileRenderCommand(const RenderCommandResolveTimings& command);
		bool CompileRenderCommand(const RenderCommandDispatch& command);
		bool CompileRenderCommand(const RenderCommandDispatchIndirect& command);
		bool CompileRenderCommand(const RenderCommandUpdateBottomLevelAS& command);
		bool CompileRenderCommand(const RenderCommandUpdateTopLevelAS& command);
		bool CompileRenderCommand(const RenderCommandTraceRays& command);
		bool CompileRenderCommand(const RenderCommandSetViewport& command);
		bool CompileRenderCommand(const RenderCommandSetScissor& command);
		bool CompileRenderCommand(const RenderCommandBeginRenderPass& command);
		bool CompileRenderCommand(const RenderCommandEndRenderPass& command);
		bool CompileRenderCommand(const RenderCommandDraw& command);
		bool CompileRenderCommand(const RenderCommandDrawIndirect& command);
	private:
		void ApplyTransitions();
		bool PrepareForDispatch(RenderBackendShaderHandle shader, const ShaderArgument& shaderArgument);
		bool PrepareForDraw(RenderBackendShaderHandle shader, PrimitiveTopology topology, RenderBackendBufferHandle indexBuffer, const ShaderArgument& shaderArgument);
		VulkanDevice* device;
		QueueFamily queueFamily;
		ID3D12GraphicsCommandList6* commandList;
		VkRenderPass activeRenderPass;
		VkPipeline activeComputePipeline;
		VkPipeline activeGraphicsPipeline;
		RenderStatistics statistics;
		std::vector<VkImageMemoryBarrier2KHR> imageBarriers;
		std::vector<VkBufferMemoryBarrier2KHR> bufferBarriers;
	};
}