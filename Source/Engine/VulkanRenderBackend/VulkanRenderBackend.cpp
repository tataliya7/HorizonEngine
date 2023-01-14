module;

#include <map>
#include <unordered_map>
#include <vector>
#include <queue>
#include <array>

#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

#include "Core/CoreDefinitions.h"
#include "Render/RenderDefinitions.h"

module HorizonEngine.Render.VulkanRenderBackend;

__pragma(warning(push, 0))
import <VulkanCommon.h>;
import HorizonEngine.Core; 
import HorizonEngine.Render.Core;
import :Utils;
__pragma(warning(pop))

#define VK_CHECK(VkFunction) { const VkResult result = VkFunction; if (result != VK_SUCCESS) { VerifyVkResult(result, #VkFunction, __FILE__, __LINE__); } }
#define VK_CHECK_RESULT(result) { if (result != VK_SUCCESS) { VerifyVkResult(result, __FUNCTION__, __FILE__, __LINE__); } }

#define VULKAN_RENDER_BACKEND_BINDLESS_DESCRIPTOR_SLOT_SAMPLED_IMAGES          0
#define VULKAN_RENDER_BACKEND_BINDLESS_DESCRIPTOR_SLOT_SAMPLERS                1
#define VULKAN_RENDER_BACKEND_BINDLESS_DESCRIPTOR_SLOT_STORAGE_IMAGES          2
#define VULKAN_RENDER_BACKEND_BINDLESS_DESCRIPTOR_SLOT_STORAGE_BUFFERS         3
#define VULKAN_RENDER_BACKEND_BINDLESS_DESCRIPTOR_SLOT_ACCELERATION_STRUCTURES 4

namespace HE
{
	namespace VulkanHelper
	{
		void CreateTemporaryCommandBuffer(VkDevice device, uint32 queueFamilyIndex, VkCommandPool& tempCmdPool, VkCommandBuffer& tempCmdBuffer)
		{
			VkCommandPoolCreateInfo commandPoolInfo = {};
			commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			commandPoolInfo.queueFamilyIndex = queueFamilyIndex;
			commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, VULKAN_ALLOCATION_CALLBACKS, &tempCmdPool));
			VkCommandBufferAllocateInfo allocateInfo = {};
			allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocateInfo.commandPool = tempCmdPool;
			allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocateInfo.commandBufferCount = 1;
			VK_CHECK(vkAllocateCommandBuffers(device, &allocateInfo, &tempCmdBuffer));
			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			VK_CHECK(vkBeginCommandBuffer(tempCmdBuffer, &beginInfo));
		}

		void FlushTemporaryCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool tempCmdPool, VkCommandBuffer tempCmdBuffer)
		{
			VK_CHECK(vkEndCommandBuffer(tempCmdBuffer));
			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &tempCmdBuffer;
			VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
			VK_CHECK(vkDeviceWaitIdle(device));
			vkDestroyCommandPool(device, tempCmdPool, nullptr);
		}
	}

struct VulkanRenderBackend;
class VulkanCommandBufferManager;

struct VulkanPushConstants
{
	uint32 indices[16];
	float data[16];
};

struct VulkanRenderBackendHandleManager
{
	std::vector<uint32> freeIndices;
	uint32 nextIndex;
	template <typename HandleType>
	HandleType Allocate(uint32 deviceMask)
	{
		uint32 index = 0;
		if (!freeIndices.empty())
		{
			index = freeIndices.back();
			freeIndices.pop_back();
		}
		else
		{
			nextIndex++;
			index = nextIndex;
		}
		HandleType handle = HandleType(index, deviceMask);
		return handle;
	}
	template <typename HandleType>
	void Free(HandleType handle)
	{
		uint32 index = ((RenderBackendHandle)handle).GetIndex();
		freeIndices.push_back(index);
	}
};

enum
{
	BindlessBindingSampledImages = 0,
	BindlessBindingSamplers = 1,
	BindlessBindingStroageImages = 2,
	BindlessBindingStroageBuffers = 3,
	BindlessBindingAccelerationStructures = 4,
};

struct VulkanBindlessConfig
{
	uint32 numSampledImages;
	uint32 numSamplers;
	uint32 numStorageImages;
	uint32 numStorageBuffers;
	uint32 numAccelerationStructures;
};

struct VulkanBindlessManager
{
	VulkanBindlessConfig config;

	VkDescriptorPool pool;
	VkDescriptorSetLayout layout;
	VkDescriptorSet set;

	std::vector<uint32> freeSampledImages;
	std::vector<uint32> freeSamplers;
	std::vector<uint32> freeStorageImages;
	std::vector<uint32> freeStorageBuffers;
	std::vector<uint32> freeAccelerationStructures;

	uint32 AllocateSampledImageIndex()
	{
		uint32 index = freeSampledImages.back();
		freeSampledImages.pop_back();
		return index;
	}

	uint32 AllocateSamplerIndex()
	{
		uint32 index = freeSamplers.back();
		freeSamplers.pop_back();
		return index;
	}

	uint32 AllocateStorageImageIndex()
	{
		uint32 index = freeStorageImages.back();
		freeStorageImages.pop_back();
		return index;
	}

	uint32 AllocateStorageBufferIndex()
	{
		uint32 index = freeStorageBuffers.back();
		freeStorageBuffers.pop_back();
		return index;
	}

	uint32 AllocateAccelerationStructureIndex()
	{
		uint32 index = freeAccelerationStructures.back();
		freeAccelerationStructures.pop_back();
		return index;
	}
};

struct VulkanPhysicalDevice
{
	VkPhysicalDevice handle;
	VkPhysicalDeviceProperties properties;
	VkPhysicalDeviceMemoryProperties memoryProperties;
	VkPhysicalDeviceDescriptorIndexingProperties descriptorIndexingProperties;
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties;
	VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties;
	VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeatures;
	VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures;
	VkPhysicalDeviceSynchronization2Features synchronization2Features;
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures;
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures;
	VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures;
	VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures shaderDemoteToHelperInvocationFeatures;
	// requiredDeviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
	/*VkPhysicalDeviceDeferredHostOperationsFeaturesKHR das;
	VkPhysicalDevicePipelineLibraryFeaturesKHR dasda;
	VkPhysicalDeviceMaintenance3FeaturesKHR synchronization2Features;*/
	void* featuresEntry;
	VkPhysicalDeviceFeatures enabledFeatures;
	std::vector<VkLayerProperties> layerProperties;
	std::vector<VkExtensionProperties> extensionProperties;
	std::vector<VkQueueFamilyProperties> queueFamilyProperties;
	uint32 queueFamilyIndices[NUM_QUEUE_FAMILIES];
};

struct VulkanQueue
{
	VkQueue handle;
	uint32 familyIndex;
	uint32 queueIndex;
};

struct VulkanSwapchain
{
	enum class Status
	{
		Success,
		OutOfDate,
		Error,
	};

	VkSwapchainKHR             handle;
	VkSurfaceKHR               surface;
	VkSwapchainCreateInfoKHR   info;
	uint32                     numBuffers;
	uint32                     activeBackBufferIndex;
	uint32                     semaphoreIndex;
	RenderBackendTextureHandle buffers[MaxNumSwapChainBuffers];
	VkFence                    imageAcquiredFences[MaxNumSwapChainBuffers];
	VkSemaphore                imageAcquiredSemaphores[MaxNumSwapChainBuffers];
};

struct VulkanCpuReadbackBuffer
{
	VkBuffer handle;
	uint32 mipOffsets[MaxNumTextureMipLevels];
	uint32 mipSize[MaxNumTextureMipLevels];
	void* data;
};

struct VulkanTexture
{
	VkImage handle;
	bool swapchainBuffer;
	VmaAllocation allocation;
	uint32 width;
	uint32 height;
	uint32 depth;
	uint32 arrayLayers;
	uint32 mipLevels;
	VkFormat format;
	VkImageType type;
	TextureType t;
	TextureCreateFlags flags;
	VkImageAspectFlags aspectMask;
	VkClearValue clearValue;
	VkImageView srv = VK_NULL_HANDLE;
	VkImageView rtv = VK_NULL_HANDLE;
	std::vector<VkImageView> dsv;
	int32 srvIndex = -1;
	struct UAV
	{
		VkImageView uav = VK_NULL_HANDLE;
		int32 uavIndex = -1;
	};
	std::vector<UAV> uavs;
	VulkanCpuReadbackBuffer* cpuReadbackBuffer;
};

struct VulkanSampler
{
	VkSampler handle;
	uint32 bindlessIndex;
};

struct VulkanRayTracingPipelineState
{
	VkPipelineLayout pipelineLayout;
	VkPipeline handle;
	uint32 numRayGenerationShaders;
	uint32 numMissShaders;
	uint32 numHitGroups;
};

struct VulkanRayTracingShaderBindingTable
{
	VkStridedDeviceAddressRegionKHR rayGenShaderBindingTable;
	VkStridedDeviceAddressRegionKHR missShaderBindingTable;
	VkStridedDeviceAddressRegionKHR hitShaderBindingTable;
	VkStridedDeviceAddressRegionKHR callableShaderBindingTable;
};

struct VulkanBuffer
{
	VkBuffer handle;
	VmaAllocation allocation;
	uint64 size;
	VkBufferUsageFlags usageFlags;
	VmaAllocationCreateFlags allocationFlags;
	VmaMemoryUsage memeryUsage;
	bool createMapped;
	bool mapped;
	void* mappedData;
	int32 uavIndex;
	std::string name;
	VulkanRayTracingShaderBindingTable* shaderBindingTable;
};


struct PipelineState
{
	VkPipelineRasterizationStateCreateInfo rasterizationState;
	VkPipelineDepthStencilStateCreateInfo depthStencilState;
	VkPipelineColorBlendStateCreateInfo colorBlendState;
	VkPipelineColorBlendAttachmentState colorBlendAttachmentStates[MaxNumSimultaneousColorRenderTargets];
	uint32 numStages;
	VkPipelineShaderStageCreateInfo stages[MaxNumShaderStages];
	std::string entryPoints[MaxNumShaderStages];
};

struct VulkanShader
{
	VkPipelineRasterizationStateCreateInfo rasterizationState;
	VkPipelineDepthStencilStateCreateInfo depthStencilState;
	VkPipelineColorBlendStateCreateInfo colorBlendState;
	VkPipelineColorBlendAttachmentState colorBlendAttachmentStates[MaxNumSimultaneousColorRenderTargets];
	uint32 numStages;
	VkPipelineShaderStageCreateInfo stages[MaxNumShaderStages]; 
	std::string entryPoints[MaxNumShaderStages];
	std::vector<uint64> pipelineHashes;
};

struct VulkanShaderCompiler
{
	MemoryArena* allocator;
};

struct VulkanRayTracingAccelerationStructure
{
	uint32 deviceMask;
	VkAccelerationStructureKHR handle;
	VkBuildAccelerationStructureFlagsKHR buildFlags;
	VkDeviceAddress deviceAddress;
	struct Buffer
	{
		VkBuffer buffer;
		uint64 size;
		VmaAllocation allocation;
		VmaAllocationInfo allocationInfo;
		VkDeviceAddress deviceAddress;
	}; 
	Buffer accelerationStructureBuffer;
	Buffer scratchBuffer;
	std::vector<Buffer> resourceBuffers;
	std::vector<VkAccelerationStructureGeometryKHR> geometries;
	union
	{
		RenderBackendBottomLevelASDesc blasDesc;
		RenderBackendTopLevelASDesc tlasDesc;
	};
	uint32 descriptorIndex;
};

struct VulkanPipeline
{
	uint64 hash;
	VkPipeline handle;
	VkPipelineLayout layout;
};

struct VulkanPipelineManager
{
	VkPipelineCache pipelineCache;
	std::unordered_map<uint64, VulkanPipeline> pipelineMap;
	std::vector<VulkanPipeline> pipelines;
	std::unordered_map<uint64, VkPipelineLayout> pipelineLayoutMap;
	std::vector<VkPipelineLayout> pipelineLayouts;
};

struct VulkanCommandBuffer
{
	VkCommandBuffer handle;
	VkFence fence;
	VkSemaphore semaphore;
};

struct VulkanSubmitContext
{
	VkSemaphore completeSemaphore;
	VkFence completeFence;
	uint32 numPrimaryCommandBuffers;
	uint32 numSecondaryCommandBuffers;
};

struct VulkanFramebuffer
{
	VkFramebuffer handle;
	uint32 width;
	uint32 height;
	uint32 layers;
	uint32 numAttachments;
	uint32 numColorAttachments;
	VkImage images[MaxNumSimultaneousColorRenderTargets + 1];
	VkImageView attachments[MaxNumSimultaneousColorRenderTargets + 1];
};

struct VulkanRenderPassDesc
{
	VkExtent3D extent;
	bool hasDepthStencil;
	uint32 renderPassCompatibleHash;
	uint32 renderPassFullHash;
	uint32 numAttachmentDescriptions;
	uint32 numColorAttachments;
	VkAttachmentDescription attachmentDescriptions[MaxNumSimultaneousColorRenderTargets + 1];
	VkAttachmentReference colorReferences[MaxNumSimultaneousColorRenderTargets];
	VkAttachmentReference depthStencilReference;
	VkImageLayout depthStencilLayout;
};

struct VulkanTimingQueryHeap
{
	VkQueryPool handle;
	uint32 maxQueryCount;
};

struct VulkanOcclusionQueryHeap
{
	VkQueryPool handle;
	uint32 maxQueryCount;
};

struct RenderPassCompatibleHashInfo
{
	uint8 numAttachments;
	VkFormat formats[MaxNumSimultaneousColorRenderTargets + 1];
};

struct RenderPassFullHashInfo
{
	/** +2 : 1 for depth, 1 for stencil. */
	uint8 loadOps[MaxNumSimultaneousColorRenderTargets + 2];
	uint8 storeOps[MaxNumSimultaneousColorRenderTargets + 2];
};

struct ResourceToDestroy
{
	enum class Type
	{
		Buffer,
		Texture,
		Sampler,
	};
	Type type;
	uint64 vkHandle;
	VmaAllocation allocation;
	int32 bindlessSRV;
	int32 bindlessUAV;
};

class VulkanDevice
{
public:
	VulkanDevice();
	~VulkanDevice();
	bool Init(VulkanRenderBackend* backend, VulkanPhysicalDevice* physicalDevive, const VulkanBindlessConfig& bindlessConfig);
	void Shutdown();
	void Tick();
	void WaitIdle();
	bool IsDeviceExtensionEnabled(const char* extension);
	void ResizeSwapChain(uint32 index, uint32* width, uint32* height);
	VulkanSwapchain::Status AcquireImageIndex(uint32 index);
	VulkanSwapchain::Status PresentSwapChain(uint32 index, VkSemaphore* waitSemaphores, uint32 waitSemaphoreCount);
	void RecreateSwapChain(uint32 index); 
	RenderBackendTextureHandle GetActiveSwapChainBackBuffer(uint32 index);
	uint32 CreateSwapChain(uint64 window);
	void DestroySwapChain(uint32 index);
	uint32 CreateBuffer(const RenderBackendBufferDesc* desc, const char* name);
	void DestroyBuffer(uint32 index);
	void ResizeBuffer(uint32 index, uint64 size);
	uint64 GetBufferDeviceAddress(RenderBackendBufferHandle bufferHandle);
	void* MapBuffer(uint32 index);
	void UnmapBuffer(uint32 index);
	uint32 CreateTexture(const RenderBackendTextureDesc* desc, const void* data, const char* name);
	void DestroyTexture(uint32 index);
	uint32 CreateTextureSRV(uint32 textureIndex, const RenderBackendTextureSRVDesc* desc, const char* name);
	int32 GetTextureSRVDescriptorIndex(uint32 textureIndex);
	uint32 CreateTextureUAV(uint32 textureIndex, const RenderBackendTextureUAVDesc* desc, const char* name);
	int32 GetTextureUAVDescriptorIndex(uint32 textureIndex, uint32 mipLevel);
	uint32 CreateSampler(const RenderBackendSamplerDesc* desc, const char* name);
	void DestroySampler(uint32 index);
	uint32 CreateShader(const RenderBackendShaderDesc* desc, const char* name);
	void DestroyShader(uint32);
	uint32 CreateBottomLevelAS(const RenderBackendBottomLevelASDesc* desc, const char* name);
	uint32 CreateTopLevelAS(const RenderBackendTopLevelASDesc* desc, const char* name);
	VkRenderPass FindOrCreateRenderPass(const VulkanRenderPassDesc& renderPassDesc);
	VulkanFramebuffer* FindOrCreateFramebuffer(const RenderPassInfo& renderPassInfo, const VulkanRenderPassDesc& renderPassDesc, VkRenderPass renderPass);
	VkPipelineLayout FindOrCreatePipelineLayout(uint32 pushConstantSize, RenderBackendPipelineType pipelineType);
	VulkanPipeline* FindOrCreateComputePipeline(VulkanShader* shader, uint32 pushConstantSize);
	VulkanPipeline* FindOrCreateRayTracingPipeline(VulkanShader* shader, uint32 pushConstantSize);
	VulkanPipeline* FindOrCreateGraphicsPipeline(VulkanShader* shader, VkRenderPass renderPass, PrimitiveTopology topology, uint32 pushConstantSize);
	void SetDebugUtilsObjectName(VkObjectType type, uint64 handle, const char* name); 
	inline VkDevice GetHandle() const
	{
		return handle;
	}
	inline VulkanRenderBackend* GetBackend()
	{
		return backend;
	}
	inline uint32 GetDeviceMask() const
	{
		return deviceMask;
	}
	inline VkDescriptorSet GetBindlessGlobalSet() const
	{
		return bindlessManager.set;
	}
	inline uint32 GetQueueFamilyIndex(QueueFamily family) const 
	{
		return physicalDevice->queueFamilyIndices[(uint32)family];
	}
	inline VulkanQueue* GetCommandQueue(uint32 family, uint32 index)
	{
		return &commandQueues[family].at(index);
	}
	inline VulkanQueue* GetCommandQueue(QueueFamily family, uint32 index)
	{
		return &commandQueues[(uint32)family].at(index);
	}
	inline const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& GetRayTracingPipelineProperties() const
	{
		return physicalDevice->rayTracingPipelineProperties;
	}
	inline void GetRenderStatistics(RenderStatistics* statistics)
	{
		*statistics = renderStatistics;
	}
	inline VulkanTexture* GetTexture(RenderBackendTextureHandle handle)
	{
		uint32 index = GetRenderBackendHandleRepresentation(handle.GetIndex());
		return &textures[index];
	}
	inline VulkanBuffer* GetBuffer(RenderBackendBufferHandle handle)
	{
		uint32 index = GetRenderBackendHandleRepresentation(handle.GetIndex());
		return &buffers[index];
	}
	inline VulkanSampler* GetSampler(RenderBackendSamplerHandle handle)
	{
		uint32 index = GetRenderBackendHandleRepresentation(handle.GetIndex());
		return &samplers[index];
	}
	inline VulkanShader* GetShader(RenderBackendShaderHandle handle)
	{
		uint32 index = GetRenderBackendHandleRepresentation(handle.GetIndex());
		return &shaders[index];
	}
	inline const VulkanTimingQueryHeap& GetTimingQueryHeap(RenderBackendTimingQueryHeapHandle handle)
	{
		uint32 index = GetRenderBackendHandleRepresentation(handle.GetIndex());
		return timingQueryHeaps.Get(index);
	}
	inline VulkanRayTracingAccelerationStructure* GetAccelerationStructure(RenderBackendRayTracingAccelerationStructureHandle handle)
	{
		uint32 index = GetRenderBackendHandleRepresentation(handle.GetIndex());
		return &accelerationStructures[index];
	}
	inline VulkanRayTracingPipelineState* GetRayTracingPipelineState(RenderBackendRayTracingPipelineStateHandle handle)
	{
		uint32 index = GetRenderBackendHandleRepresentation(handle.GetIndex());
		return &rayTracingPipelineStates[index];
	}
	inline uint32 GetRenderBackendHandleRepresentation(uint32 handle)
	{
		ASSERT(handleRepresentations.find(handle) != handleRepresentations.end());
		return handleRepresentations[handle];
	}
	inline bool TryGetRenderBackendHandleRepresentation(uint32 handle, uint32* outValue)
	{
		if (handleRepresentations.find(handle) == handleRepresentations.end())
		{
			return false;
		}
		*outValue = handleRepresentations[handle];
		return true;
	}
	inline void SetRenderBackendHandleRepresentation(uint32 handle, uint32 value)
	{
		handleRepresentations[handle] = value;
	}
	inline bool RemoveRenderBackendHandleRepresentation(uint32 handle)
	{
		return handleRepresentations.erase(handle);
	}
	std::vector<VkSemaphore> renderCompleteSemaphores;
	VulkanCommandBufferManager* commandBufferManager;
	RenderStatistics renderStatistics;
	std::vector<VulkanSwapchain> swapchains;
	void CreateVmaAllocator();
	void DestroyVmaAllocator();
	bool CreateBindlessManager(const VulkanBindlessConfig& bindlessConfig);
	void DestroyBindlessManager();
	void CreateDefaultResources(); 
	uint32 CreateAccelerationStructure(VulkanRayTracingAccelerationStructure* accelerationStructure, VkAccelerationStructureTypeKHR type, uint32* primitiveCounts, const char* name);
	MemoryArena*          allocator;
	VulkanRenderBackend*  backend;
	VulkanPhysicalDevice* physicalDevice;
	VkInstance            instance;
	VkDevice              handle;
	VmaAllocator          vmaAllocator;
	uint32                deviceMask;

	std::vector<const char*> enabledDeviceExtensions;
	std::vector<const char*> enabledValidationLayers;

	uint32 numCommandQueues[NUM_QUEUE_FAMILIES] = { 1, 1, 1 };
	std::vector<VulkanQueue> commandQueues[NUM_QUEUE_FAMILIES];

	RenderBackendSamplerHandle defaultSampler;
	RenderBackendBufferHandle defaultStorageBuffer;

	VulkanBindlessManager bindlessManager;
	VulkanPipelineManager pipelineManager;

	struct FramebufferList
	{
		std::vector<VulkanFramebuffer> framebuffers;
	};
	std::map<uint32, FramebufferList> cachedFramebuffers;
	std::map<uint32, VkRenderPass> cachedRenderPasses;

	template<typename ResourceType>
	struct ResourceContainer
	{
		uint32 Add(const ResourceType& resource)
		{
			uint32 index = 0;
			if (!freeResourceIndices.empty())
			{
				index = freeResourceIndices.back();
				freeResourceIndices.pop_back();
				resources[index] = resource;
			}
			else
			{
				index = (uint32)resources.size();
				resources.emplace_back(resource);
			}
			return index;
		}
		void Free(uint32 index)
		{
			freeResourceIndices.push_back(index);
		}
		const ResourceType& Get(uint32 index)
		{
			return resources[index];
		}
		std::vector<ResourceType> resources;
		std::vector<uint32> freeResourceIndices;
	};

	std::vector<VulkanBuffer> buffers;
	std::vector<uint32> freeBuffers;
	std::vector<VulkanTexture> textures;
	std::vector<uint32> freeTextures;
	std::vector<VulkanSampler> samplers;
	std::vector<uint32> freeSamplers;
	std::vector<VulkanShader> shaders;
	std::vector<uint32> freeShaders;
	ResourceContainer<VulkanTimingQueryHeap> timingQueryHeaps;
	ResourceContainer<VulkanOcclusionQueryHeap> occlusionQueryHeaps;

	std::vector<VulkanRayTracingAccelerationStructure> accelerationStructures;
	std::vector<uint32> freeAccelerationStructures;
	std::vector<VulkanRayTracingPipelineState> rayTracingPipelineStates;

	std::queue<ResourceToDestroy> resourcesToDestroy;

	std::map<uint32, uint32> handleRepresentations;
};

class VulkanCommandBufferManager
{
public:
	VulkanCommandBufferManager(VulkanDevice* device, QueueFamily family)
		: device(device)
		, queueFamily(family)
	{
		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = device->GetQueueFamilyIndex(family);
		VK_CHECK(vkCreateCommandPool(device->GetHandle(), &poolInfo, VULKAN_ALLOCATION_CALLBACKS, &pool));
	}
	~VulkanCommandBufferManager()
	{
		vkDestroyCommandPool(device->GetHandle(), pool, VULKAN_ALLOCATION_CALLBACKS);
		pool = VK_NULL_HANDLE;
	}
	inline VkCommandPool GetCommandPoolHandle() const
	{
		return pool;
	}
	VulkanCommandBuffer* AllocateCommandBuffer()
	{
		VulkanCommandBuffer commandBuffer;
		VkCommandBufferAllocateInfo allocateInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		VK_CHECK(vkAllocateCommandBuffers(device->GetHandle(), &allocateInfo, &commandBuffer.handle));
		VkFenceCreateInfo fenceInfo = {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};
		VK_CHECK(vkCreateFence(device->GetHandle(), &fenceInfo, VULKAN_ALLOCATION_CALLBACKS, &commandBuffer.fence));
		VkSemaphoreCreateInfo semaphoreInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};
		VK_CHECK(vkCreateSemaphore(device->GetHandle(), &semaphoreInfo, VULKAN_ALLOCATION_CALLBACKS, &commandBuffer.semaphore));
		commandBuffers.emplace_back(commandBuffer);
		return &commandBuffers.back();
	}
	VulkanCommandBuffer* PrepareForNextCommandBuffer()
	{
		for (VulkanCommandBuffer& commandBuffer : commandBuffers)
		{
			VkResult result = vkGetFenceStatus(device->GetHandle(), commandBuffer.fence);
			switch (result)
			{
			case VK_SUCCESS:
				return &commandBuffer;
			case VK_NOT_READY:
				break;
			default:
				VK_CHECK_RESULT(result);
				break;
			}
		}
		return AllocateCommandBuffer();
	}
private:
	VulkanDevice* device;
	QueueFamily queueFamily;
	VkCommandPool pool;
	std::vector<VulkanCommandBuffer> commandBuffers;
};

struct VulkanRenderBackend
{
	VkInstance instance;
	std::vector<const char*> enabledInstanceLayers;
	std::vector<const char*> enabledInstanceExtensions;
	VkDebugUtilsMessengerEXT debugUtilsMessenger;
	uint32 numDevices;
	VulkanDevice devices[MaxNumRenderDevices];
	VulkanPhysicalDevice availablePhysicalDevices[MaxNumRenderDevices];
	VulkanRenderBackendHandleManager handleManager;
	struct VulkanFunctions
	{
		PFN_vkSetDebugUtilsObjectNameEXT               vkSetDebugUtilsObjectNameEXT;
		PFN_vkCmdPipelineBarrier2KHR                   vkCmdPipelineBarrier2KHR;
		PFN_vkGetBufferDeviceAddressKHR                vkGetBufferDeviceAddressKHR;
		PFN_vkCreateAccelerationStructureKHR           vkCreateAccelerationStructureKHR;
		PFN_vkDestroyAccelerationStructureKHR          vkDestroyAccelerationStructureKHR;
		PFN_vkGetAccelerationStructureBuildSizesKHR    vkGetAccelerationStructureBuildSizesKHR;
		PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
		PFN_vkGetRayTracingShaderGroupHandlesKHR       vkGetRayTracingShaderGroupHandlesKHR;
		PFN_vkBuildAccelerationStructuresKHR           vkBuildAccelerationStructuresKHR;
		PFN_vkCreateRayTracingPipelinesKHR             vkCreateRayTracingPipelinesKHR;
		PFN_vkCmdBuildAccelerationStructuresKHR        vkCmdBuildAccelerationStructuresKHR;
		PFN_vkCmdTraceRaysKHR                          vkCmdTraceRaysKHR;
	};
	VulkanFunctions functions;
	bool Init(int flags);
	void Exit();
	void EnumeratePhysicalDevices();
	bool IsInstanceExtensionEnabled(const char* extension);
};

VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsMessengerCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType, 
	const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
	void* userData)
{
	if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
	{
		HE_LOG_WARNING("{} - {}: {}", callbackData->messageIdNumber, callbackData->pMessageIdName, callbackData->pMessage);
	}
	else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
	{
		HE_LOG_ERROR("{} - {}: {}", callbackData->messageIdNumber, callbackData->pMessageIdName, callbackData->pMessage);
	}
	return VK_FALSE;
}

static bool CheckInstanceExtensionSupport(const char* name, const std::vector<VkExtensionProperties>& supportedExtensions)
{
	for (const auto& supportedExtension : supportedExtensions)
	{
		if (strcmp(name, supportedExtension.extensionName) == 0)
		{
			return true;
		}
	}
	return false;
}

static bool CheckInstanceLayerSupport(const char* name, const std::vector<VkLayerProperties>& supportedLayers)
{
	for (const auto& supportedLayer : supportedLayers)
	{
		if (strcmp(name, supportedLayer.layerName) == 0)
		{
			return true;
		}
	}
	return false;
}

bool VulkanRenderBackend::IsInstanceExtensionEnabled(const char* extension)
{
	for (const auto& enabledExtension : enabledInstanceExtensions)
	{
		if (strcmp(extension, enabledExtension) == 0)
		{
			return true;
		}
	}
	return false;
}

void VulkanRenderBackend::EnumeratePhysicalDevices()
{
	VK_CHECK(vkEnumeratePhysicalDevices(instance, &numDevices, 0));
	if (numDevices == 0)
	{
		HE_LOG_INFO("No available physical device.\n");
		return;
	}
	VkPhysicalDevice physicalDeviceHandles[MaxNumRenderDevices];
	ASSERT(numDevices < MaxNumRenderDevices);
	vkEnumeratePhysicalDevices(instance, &numDevices, physicalDeviceHandles);

	for (uint32 index = 0; index < numDevices; index++)
	{
		VulkanPhysicalDevice& physicalDevice = availablePhysicalDevices[index];

		physicalDevice.handle = physicalDeviceHandles[index];

		VkPhysicalDeviceProperties2 physicalDeviceProperties2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext = &physicalDevice.descriptorIndexingProperties,
		};
		physicalDevice.descriptorIndexingProperties = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES_EXT,
			.pNext = &physicalDevice.rayTracingPipelineProperties
		};
		physicalDevice.rayTracingPipelineProperties = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
			.pNext = &physicalDevice.accelerationStructureProperties,
		};
		physicalDevice.accelerationStructureProperties = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR
		};
		vkGetPhysicalDeviceProperties2(physicalDevice.handle, &physicalDeviceProperties2);
		physicalDevice.properties = physicalDeviceProperties2.properties;

		vkGetPhysicalDeviceMemoryProperties(physicalDevice.handle, &physicalDevice.memoryProperties);

		// Ray tracing features
		physicalDevice.accelerationStructureFeatures = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
			.pNext = &physicalDevice.rayTracingPipelineFeatures
		};
		physicalDevice.rayTracingPipelineFeatures = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
			.pNext = &physicalDevice.rayQueryFeatures
		};
		physicalDevice.rayQueryFeatures = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
			.pNext = &physicalDevice.shaderDemoteToHelperInvocationFeatures
		};
		physicalDevice.shaderDemoteToHelperInvocationFeatures = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES,
			.pNext = &physicalDevice.bufferDeviceAddressFeatures
		};
		physicalDevice.bufferDeviceAddressFeatures = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
			.pNext = &physicalDevice.descriptorIndexingFeatures
		};
		physicalDevice.descriptorIndexingFeatures = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT,
			.pNext = &physicalDevice.synchronization2Features
		};
		physicalDevice.synchronization2Features = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
			.pNext = nullptr
		};

#if DEBUG_ONLY_RAY_TRACING_ENBALE
		physicalDevice.featuresEntry = (void*)&physicalDevice.accelerationStructureFeatures;
#else
		physicalDevice.featuresEntry = (void*)&physicalDevice.bufferDeviceAddressFeatures;
#endif

		VkPhysicalDeviceFeatures2 deviceFeatures2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
			.pNext = physicalDevice.featuresEntry
		};
		vkGetPhysicalDeviceFeatures2(physicalDevice.handle, &deviceFeatures2);
		physicalDevice.enabledFeatures = deviceFeatures2.features;

		HE_LOG_INFO("Found physical device (name: {}, type: {}, vendor id: {}, device id: {}, support vulkan version: {}.{}.{})",
			physicalDevice.properties.deviceName,
			(int32)physicalDevice.properties.deviceType,
			physicalDevice.properties.vendorID,
			physicalDevice.properties.deviceID,
			VK_API_VERSION_MAJOR(physicalDevice.properties.apiVersion),
			VK_API_VERSION_MINOR(physicalDevice.properties.apiVersion),
			VK_API_VERSION_PATCH(physicalDevice.properties.apiVersion)
		);

		uint32 numQueueFamilyProperties;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice.handle, &numQueueFamilyProperties, 0);
		physicalDevice.queueFamilyProperties.resize(numQueueFamilyProperties);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice.handle, &numQueueFamilyProperties, physicalDevice.queueFamilyProperties.data());

		uint32& graphicsQueueFamilyIndex = physicalDevice.queueFamilyIndices[(uint32)QueueFamily::Graphics] = ~uint32(0);
		uint32& computeQueueFamilyIndex = physicalDevice.queueFamilyIndices[(uint32)QueueFamily::Compute] = ~uint32(0);
		uint32& transferQueueFamilyIndex = physicalDevice.queueFamilyIndices[(uint32)QueueFamily::Copy] = ~uint32(0);

		for (uint32 i = 0; i < numQueueFamilyProperties; i++)
		{
			VkQueueFlags queueFlags = physicalDevice.queueFamilyProperties[i].queueFlags;
			if ((queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 && graphicsQueueFamilyIndex == ~uint32(0))
			{
				graphicsQueueFamilyIndex = i;
			}
			else if ((queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 && computeQueueFamilyIndex == ~uint32(0))
			{
				computeQueueFamilyIndex = i;
			}
			else if ((queueFlags & VK_QUEUE_TRANSFER_BIT) != 0 && transferQueueFamilyIndex == ~uint32(0))
			{
				transferQueueFamilyIndex = i;
			}
		}

		uint32 numLayerProperties = 0;
		VK_CHECK(vkEnumerateDeviceLayerProperties(physicalDevice.handle, &numLayerProperties, nullptr));
		physicalDevice.layerProperties.resize(numLayerProperties);
		VK_CHECK(vkEnumerateDeviceLayerProperties(physicalDevice.handle, &numLayerProperties, physicalDevice.layerProperties.data()));
		for (const auto& layerProperties : physicalDevice.layerProperties)
		{
			HE_LOG_INFO("Available device layer: {} - vulkan apid version: {}.{}.{} - implemetation version: {} - description: {}.",
				layerProperties.layerName,
				VK_API_VERSION_MAJOR(layerProperties.specVersion),
				VK_API_VERSION_MINOR(layerProperties.specVersion),
				VK_API_VERSION_PATCH(layerProperties.specVersion),
				layerProperties.implementationVersion,
				layerProperties.description
			);
		}

		uint32 numExtensionProperties = 0;
		VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice.handle, nullptr, &numExtensionProperties, nullptr));
		physicalDevice.extensionProperties.resize(numExtensionProperties);
		VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice.handle, nullptr, &numExtensionProperties, physicalDevice.extensionProperties.data()));
		for (const auto& extensionProperty : physicalDevice.extensionProperties)
		{
			HE_LOG_INFO("Available device extension: {} - extension version: {}.",
				extensionProperty.extensionName,
				extensionProperty.specVersion
			);
		}
	}
}

bool VulkanRenderBackend::Init(int flags)
{
	bool enableValidationLayers = flags & VULKAN_RENDER_BACKEND_CREATE_FLAGS_VALIDATION_LAYERS;

	std::vector<const char*> requiredInstanceLayers;
	if (enableValidationLayers)
	{
		requiredInstanceLayers.push_back("VK_LAYER_KHRONOS_validation");
		requiredInstanceLayers.push_back("VK_LAYER_KHRONOS_synchronization2");
		// requiredInstanceLayers.push_back("VK_LAYER_KHRONOS_profiles");
	}

	uint32 numInstanceLayerProperties = 0;
	VK_CHECK(vkEnumerateInstanceLayerProperties(&numInstanceLayerProperties, nullptr));
	std::vector<VkLayerProperties> instanceLayerProperties(numInstanceLayerProperties);
	VK_CHECK(vkEnumerateInstanceLayerProperties(&numInstanceLayerProperties, instanceLayerProperties.data()));
	for (const auto& instanceLayerProperty : instanceLayerProperties)
	{
		HE_LOG_INFO("Available instance layer: {} - vulkan api version: {}.{}.{} - implementation version: {} - description: {}.",
			instanceLayerProperty.layerName,
			VK_API_VERSION_MAJOR(instanceLayerProperty.specVersion),
			VK_API_VERSION_MINOR(instanceLayerProperty.specVersion),
			VK_API_VERSION_PATCH(instanceLayerProperty.specVersion),
			instanceLayerProperty.implementationVersion,
			instanceLayerProperty.description
		);
	}

	for (const auto& requiredInstanceLayer : requiredInstanceLayers)
	{
		if (CheckInstanceLayerSupport(requiredInstanceLayer, instanceLayerProperties))
		{
			enabledInstanceLayers.push_back(requiredInstanceLayer);
			HE_LOG_INFO("Enabled instance layer: {}.", requiredInstanceLayer);
		}
		else
		{
			HE_LOG_ERROR("Missing required instance layer: {}.", requiredInstanceLayer);
			return false;
		}
	}

	uint32 numInstanceExtensionProperties = 0;
	VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &numInstanceExtensionProperties, nullptr));
	std::vector<VkExtensionProperties> instanceExtensionProperties(numInstanceExtensionProperties);
	VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &numInstanceExtensionProperties, instanceExtensionProperties.data()));
	for (const auto& instanceExtensionProperty : instanceExtensionProperties)
	{
		HE_LOG_INFO("Available instance extension: {} - extension version: {}.",
			instanceExtensionProperty.extensionName,
			instanceExtensionProperty.specVersion
		);
	}

	std::vector<const char*> requiredInstanceExtensions;
	if (enableValidationLayers)
	{
		requiredInstanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	if (flags & VULKAN_RENDER_BACKEND_CREATE_FLAGS_SURFACE)
	{
		requiredInstanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#if defined(VK_USE_PLATFORM_WIN32_KHR)
		requiredInstanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XCB_KHR) 
		requiredInstanceExtensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
		requiredInstanceExtensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#endif
	}
	requiredInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

	for (const auto& requiredInstanceExtension : requiredInstanceExtensions)
	{
		if (CheckInstanceExtensionSupport(requiredInstanceExtension, instanceExtensionProperties))
		{
			enabledInstanceExtensions.push_back(requiredInstanceExtension);
			HE_LOG_INFO("Enabled instance extension: {}.", requiredInstanceExtension);
		}
		else
		{
			HE_LOG_ERROR("Missing required instance extension: {}.", requiredInstanceExtension);
			return false;
		}
	}

	VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerInfo = {
			.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
			.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
			.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
			.pfnUserCallback = DebugUtilsMessengerCallback
	};

	VkApplicationInfo applicationInfo = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		//.pApplicationName = ,
		//.applicationVersion = ,
		.pEngineName = HE_ENGINE_NAME,
		.engineVersion = HE_ENGINE_VERSION,
		.apiVersion = VK_API_VERSION_1_3,
	};
	VkInstanceCreateInfo instanceInfo = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pNext = enableValidationLayers ? &debugUtilsMessengerInfo : nullptr,
		.pApplicationInfo = &applicationInfo,
		.enabledLayerCount = (uint32)(enabledInstanceLayers.size()),
		.ppEnabledLayerNames = enabledInstanceLayers.data(),
		.enabledExtensionCount = (uint32)(enabledInstanceExtensions.size()),
		.ppEnabledExtensionNames = enabledInstanceExtensions.data()
	};

	VkResult result = vkCreateInstance(&instanceInfo, VULKAN_ALLOCATION_CALLBACKS, &instance);
	if (result != VK_SUCCESS)
	{
		return false;
	}

	if (enableValidationLayers)
	{
		PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
		VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance, &debugUtilsMessengerInfo, VULKAN_ALLOCATION_CALLBACKS, &debugUtilsMessenger));
	}
	else
	{
		debugUtilsMessenger = VK_NULL_HANDLE;
	}

	functions.vkSetDebugUtilsObjectNameEXT               = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT"));
	functions.vkCmdPipelineBarrier2KHR                   = reinterpret_cast<PFN_vkCmdPipelineBarrier2KHR>(vkGetInstanceProcAddr(instance, "vkCmdPipelineBarrier2KHR"));
	
	functions.vkGetBufferDeviceAddressKHR                = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(vkGetInstanceProcAddr(instance, "vkGetBufferDeviceAddressKHR"));
	
	functions.vkCreateAccelerationStructureKHR           = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetInstanceProcAddr(instance, "vkCreateAccelerationStructureKHR"));
	functions.vkDestroyAccelerationStructureKHR          = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetInstanceProcAddr(instance, "vkDestroyAccelerationStructureKHR"));
	functions.vkGetAccelerationStructureBuildSizesKHR    = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetInstanceProcAddr(instance, "vkGetAccelerationStructureBuildSizesKHR"));
	functions.vkGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetInstanceProcAddr(instance, "vkGetAccelerationStructureDeviceAddressKHR"));
	functions.vkGetRayTracingShaderGroupHandlesKHR       = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(vkGetInstanceProcAddr(instance, "vkGetRayTracingShaderGroupHandlesKHR"));
	functions.vkBuildAccelerationStructuresKHR           = reinterpret_cast<PFN_vkBuildAccelerationStructuresKHR>(vkGetInstanceProcAddr(instance, "vkBuildAccelerationStructuresKHR"));
	functions.vkCreateRayTracingPipelinesKHR             = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(vkGetInstanceProcAddr(instance, "vkCreateRayTracingPipelinesKHR"));
	functions.vkCmdBuildAccelerationStructuresKHR        = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetInstanceProcAddr(instance, "vkCmdBuildAccelerationStructuresKHR"));
	functions.vkCmdTraceRaysKHR                          = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetInstanceProcAddr(instance, "vkCmdTraceRaysKHR"));

	EnumeratePhysicalDevices();

	return true;
}

void VulkanRenderBackend::Exit()
{
	if (debugUtilsMessenger != VK_NULL_HANDLE)
	{
		PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
		vkDestroyDebugUtilsMessengerEXT(instance, debugUtilsMessenger, VULKAN_ALLOCATION_CALLBACKS);
		debugUtilsMessenger = VK_NULL_HANDLE;
	}
	if (instance != VK_NULL_HANDLE)
	{
		vkDestroyInstance(instance, VULKAN_ALLOCATION_CALLBACKS);
		instance = VK_NULL_HANDLE;
	}
}

static void GetRenderPassDescAndClearValues(VulkanDevice* device, const RenderPassInfo& renderPassInfo, VkImageLayout depthStencilLayout, VulkanRenderPassDesc* outRenderPassDesc, VkClearValue* outClearValues)
{
	RenderPassCompatibleHashInfo compatibleHashInfo = {};
	RenderPassFullHashInfo fullHashInfo = {};
	bool bSetExtent = true;

	for (uint32 index = 0; index < MaxNumSimultaneousColorRenderTargets; index++)
	{
		const RenderPassInfo::ColorRenderTarget& colorRenderTarget = renderPassInfo.colorRenderTargets[index];

		if (!colorRenderTarget.texture)
		{
			continue;
		}

		VulkanTexture* texture = device->GetTexture(colorRenderTarget.texture);
		uint32 mipLevel = renderPassInfo.colorRenderTargets[index].mipLevel;

		if (bSetExtent)
		{
			outRenderPassDesc->extent.width = std::max(1u, texture->width >> mipLevel);
			outRenderPassDesc->extent.height = std::max(1u, texture->height >> mipLevel);
			outRenderPassDesc->extent.depth = texture->depth;
			bSetExtent = false;
		}
		else
		{
			ASSERT(outRenderPassDesc->extent.width == std::max(1u, texture->width >> mipLevel));
			ASSERT(outRenderPassDesc->extent.height == std::max(1u, texture->height >> mipLevel));
			ASSERT(outRenderPassDesc->extent.depth == texture->depth);
		}

		VkAttachmentDescription& attachmentDesc = outRenderPassDesc->attachmentDescriptions[outRenderPassDesc->numAttachmentDescriptions];
		attachmentDesc.samples = VK_SAMPLE_COUNT_1_BIT;
		attachmentDesc.format = texture->format;
		attachmentDesc.loadOp = ConvertToVkAttachmentLoadOp(colorRenderTarget.loadOp);
		attachmentDesc.storeOp = ConvertToVkAttachmentStoreOp(colorRenderTarget.storeOp);
		attachmentDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachmentDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachmentDesc.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachmentDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference& colorReference = outRenderPassDesc->colorReferences[outRenderPassDesc->numColorAttachments];
		colorReference.attachment = outRenderPassDesc->numAttachmentDescriptions;
		colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		compatibleHashInfo.numAttachments++;
		compatibleHashInfo.formats[outRenderPassDesc->numColorAttachments] = attachmentDesc.format;
		fullHashInfo.loadOps[outRenderPassDesc->numColorAttachments] = attachmentDesc.loadOp;
		fullHashInfo.storeOps[outRenderPassDesc->numColorAttachments] = attachmentDesc.storeOp;

		outClearValues[outRenderPassDesc->numAttachmentDescriptions].color = texture->clearValue.color;
		outRenderPassDesc->numAttachmentDescriptions++;
		outRenderPassDesc->numColorAttachments++;
	}

	if (renderPassInfo.depthStencilRenderTarget.texture)
	{
		const RenderPassInfo::DepthStencilRenderTarget& depthStencilRenderTarget = renderPassInfo.depthStencilRenderTarget;
		VulkanTexture* texture = device->GetTexture(depthStencilRenderTarget.texture);

		if (bSetExtent)
		{
			outRenderPassDesc->extent.width = texture->width;
			outRenderPassDesc->extent.height = texture->height;
			outRenderPassDesc->extent.depth = 1;
			bSetExtent = false;
		}
		else
		{
			// Depth can be greater or equal to color. Clamp to the smaller size.
			outRenderPassDesc->extent.width = std::max(outRenderPassDesc->extent.width, texture->width);
			outRenderPassDesc->extent.height = std::min(outRenderPassDesc->extent.height, texture->height);
		}

		VkAttachmentDescription& attachmentDesc = outRenderPassDesc->attachmentDescriptions[outRenderPassDesc->numAttachmentDescriptions];
		attachmentDesc.samples = VK_SAMPLE_COUNT_1_BIT;
		attachmentDesc.format = texture->format;
		attachmentDesc.loadOp = ConvertToVkAttachmentLoadOp(depthStencilRenderTarget.depthLoadOp);
		attachmentDesc.storeOp = ConvertToVkAttachmentStoreOp(depthStencilRenderTarget.depthStoreOp);
		attachmentDesc.stencilLoadOp = ConvertToVkAttachmentLoadOp(depthStencilRenderTarget.stencilLoadOp);
		attachmentDesc.stencilStoreOp = ConvertToVkAttachmentStoreOp(depthStencilRenderTarget.stencilStoreOp);
		attachmentDesc.initialLayout = depthStencilLayout;
		attachmentDesc.finalLayout = depthStencilLayout;

		outRenderPassDesc->depthStencilReference.attachment = outRenderPassDesc->numAttachmentDescriptions;
		outRenderPassDesc->depthStencilReference.layout = depthStencilLayout;

		compatibleHashInfo.formats[MaxNumSimultaneousColorRenderTargets] = attachmentDesc.format;
		fullHashInfo.loadOps[MaxNumSimultaneousColorRenderTargets] = attachmentDesc.loadOp;
		fullHashInfo.storeOps[MaxNumSimultaneousColorRenderTargets] = attachmentDesc.storeOp;
		fullHashInfo.loadOps[MaxNumSimultaneousColorRenderTargets + 1] = attachmentDesc.stencilLoadOp;
		fullHashInfo.storeOps[MaxNumSimultaneousColorRenderTargets + 1] = attachmentDesc.stencilStoreOp;

		outClearValues[outRenderPassDesc->numAttachmentDescriptions].depthStencil = texture->clearValue.depthStencil;
		outRenderPassDesc->hasDepthStencil = true;
		outRenderPassDesc->numAttachmentDescriptions++;
	}

	outRenderPassDesc->depthStencilLayout = depthStencilLayout;
	outRenderPassDesc->renderPassCompatibleHash = Crc32(&compatibleHashInfo, sizeof(compatibleHashInfo));
	outRenderPassDesc->renderPassFullHash = Crc32(&fullHashInfo, sizeof(fullHashInfo), outRenderPassDesc->renderPassCompatibleHash);
}

void VulkanDevice::Tick()
{
	while (!resourcesToDestroy.empty())
	{
		const auto& resource = resourcesToDestroy.front();
		switch (resource.type)
		{
		case ResourceToDestroy::Type::Buffer:
			vmaDestroyBuffer(vmaAllocator, (VkBuffer)resource.vkHandle, resource.allocation);
			break;
		default:
			break;
		}
		resourcesToDestroy.pop();
	}
}

uint32 VulkanDevice::CreateBuffer(const RenderBackendBufferDesc* desc, const char* name)
{
	VulkanBuffer buffer = {
		.size = desc->size,
		.usageFlags = GetVkBufferUsageFlags(desc->flags),
		.name = name,
	};

	VkBufferCreateInfo bufferInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = buffer.size,
		.usage = buffer.usageFlags,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	buffer.allocationFlags = 0;
	buffer.createMapped = false;
	if (HAS_ANY_FLAGS(desc->flags, BufferCreateFlags::CreateMapped))
	{
		buffer.allocationFlags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
		buffer.createMapped = true;
	}
	buffer.memeryUsage = GetVmaMemoryUsage(desc->flags);

	VmaAllocationCreateInfo memoryInfo = {
		.flags = buffer.allocationFlags,
		.usage = buffer.memeryUsage,
	};

	VmaAllocationInfo allocationInfo = {};
	VK_CHECK(vmaCreateBuffer(vmaAllocator, &bufferInfo, &memoryInfo, &buffer.handle, &buffer.allocation, &allocationInfo));

	if (!buffer.name.empty())
	{
		SetDebugUtilsObjectName(VK_OBJECT_TYPE_BUFFER, (uint64)buffer.handle, buffer.name.c_str());
	}

	if (buffer.createMapped)
	{
		VK_CHECK(vmaMapMemory(vmaAllocator, buffer.allocation, &buffer.mappedData));
		buffer.mapped = true;
	}

	if (HAS_ANY_FLAGS(desc->flags, BufferCreateFlags::UnorderedAccess))
	{
		uint32 index = bindlessManager.AllocateStorageBufferIndex();
		VkDescriptorBufferInfo descriptorBufferInfo = {
			.buffer = buffer.handle,
			.offset = 0,
			.range = VK_WHOLE_SIZE
		};
		VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = bindlessManager.set,
			.dstBinding = VULKAN_RENDER_BACKEND_BINDLESS_DESCRIPTOR_SLOT_STORAGE_BUFFERS,
			.dstArrayElement = index,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &descriptorBufferInfo,
		};
		vkUpdateDescriptorSets(handle, 1, &write, 0, nullptr);
		buffer.uavIndex = index;
	}

	uint32 bufferIndex = 0;
	if (!freeBuffers.empty())
	{
		bufferIndex = freeBuffers.back();
		freeBuffers.pop_back();
		buffers[bufferIndex] = buffer;
	}
	else
	{
		bufferIndex = (uint32)buffers.size();
		buffers.emplace_back(buffer);
	}
	return bufferIndex;
}

void VulkanDevice::ResizeBuffer(uint32 index, uint64 size)
{
	VulkanBuffer& buffer = buffers[index];
	if (buffer.handle != VK_NULL_HANDLE)
	{
		ResourceToDestroy resource = {
			.type = ResourceToDestroy::Type::Buffer,
			.vkHandle = (uint64)buffer.handle,
			.allocation = buffer.allocation,
		};
		resourcesToDestroy.emplace(resource);
	}
	if (size > 0)
	{
		buffer.size = size;
		VkBufferCreateInfo bufferInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = buffer.size,
			.usage = buffer.usageFlags,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		VmaAllocationCreateFlags vmaAllocationFlags = 0;
		VmaAllocationCreateInfo memoryInfo = {
			.flags = buffer.allocationFlags,
			.usage = buffer.memeryUsage,
		};
		VmaAllocationInfo allocationInfo = {};
		VK_CHECK(vmaCreateBuffer(vmaAllocator, &bufferInfo, &memoryInfo, &buffer.handle, &buffer.allocation, &allocationInfo));
		if (!buffer.name.empty())
		{
			SetDebugUtilsObjectName(VK_OBJECT_TYPE_BUFFER, (uint64)buffer.handle, buffer.name.c_str());
		}
		VkDescriptorBufferInfo descriptorBufferInfo = {
			.buffer = buffer.handle,
			.offset = 0,
			.range = VK_WHOLE_SIZE
		};
		VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = bindlessManager.set,
			.dstBinding = VULKAN_RENDER_BACKEND_BINDLESS_DESCRIPTOR_SLOT_STORAGE_BUFFERS,
			.dstArrayElement = (uint32)buffer.uavIndex,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &descriptorBufferInfo,
		};
		vkUpdateDescriptorSets(handle, 1, &write, 0, nullptr);
	}
	else
	{
		buffer.handle = VK_NULL_HANDLE;
		buffer.allocation = VK_NULL_HANDLE;
	}
}

uint64 VulkanDevice::GetBufferDeviceAddress(RenderBackendBufferHandle bufferHandle)
{
	uint32 index = GetRenderBackendHandleRepresentation(bufferHandle.GetIndex());
	VulkanBuffer& buffer = buffers[index];
	VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = buffer.handle,
	};
	return backend->functions.vkGetBufferDeviceAddressKHR(handle, &bufferDeviceAddressInfo);
}

void VulkanDevice::DestroyBuffer(uint32 index)
{
}

void* VulkanDevice::MapBuffer(uint32 index)
{
	VulkanBuffer& buffer = buffers[index];
	if (!buffer.createMapped && !buffer.mapped && !buffer.mappedData)
	{
		VK_CHECK(vmaMapMemory(vmaAllocator, buffer.allocation, &buffer.mappedData));
		buffer.mapped = true;
	}
	return buffer.mappedData;
}

void VulkanDevice::UnmapBuffer(uint32 index)
{
	VulkanBuffer& buffer = buffers[index];
	VK_CHECK(vmaFlushAllocation(vmaAllocator, buffer.allocation, 0, buffer.size));
	if (!buffer.createMapped || buffer.mapped || buffer.mappedData)
	{
		vmaUnmapMemory(vmaAllocator, buffer.allocation);
		buffer.mapped = false;
		buffer.mappedData = nullptr;
	}
}

uint32 VulkanDevice::CreateTexture(const RenderBackendTextureDesc* desc, const void* data, const char* name)
{
	uint32 textureIndex = 0;
	if (!freeTextures.empty())
	{
		textureIndex = freeTextures.back();
		freeTextures.pop_back();
	}
	else
	{
		textureIndex = (uint32)textures.size();
		textures.emplace_back();
	}

	VkFormat format = ConvertToVkFormat(desc->format);

	VulkanTexture& texture = textures[textureIndex];
	texture = {
		.swapchainBuffer = false,
		.width = desc->width,
		.height = desc->height,
		.depth = desc->depth,
		.arrayLayers = desc->arrayLayers,
		.mipLevels = desc->mipLevels,
		.format = format,
		.type = ConvertToVkImageType(desc->type),
		.t = desc->type,
		.flags = desc->flags,
		.aspectMask = GetVkImageAspectFlags(format),
		.clearValue = *(VkClearValue*)&desc->clearValue,
	};

	if (HAS_ANY_FLAGS(desc->flags, TextureCreateFlags::Readback))
	{
		texture.cpuReadbackBuffer = new VulkanCpuReadbackBuffer();
		
		uint32 stride = GPixelFormatTable[(uint32)desc->format].bytes;
		uint32 width = texture.width;
		uint32 height = texture.height;
		uint32 depth = texture.depth;
		uint64 size = 0;
		for (uint32 mipLevel = 0; mipLevel < desc->mipLevels; mipLevel++)
		{
			uint32 mipSize = width * height * depth * stride;
			texture.cpuReadbackBuffer->mipOffsets[mipLevel] = size;
			texture.cpuReadbackBuffer->mipSize[mipLevel] = mipSize;
			width = std::max(1u, width / 2);
			height = std::max(1u, height / 2);
			depth = std::max(1u, depth / 2);
			size += mipSize;
		}

		VkBufferCreateInfo bufferInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		};

		VmaAllocationCreateInfo memoryInfo = {
			.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
			.usage = VMA_MEMORY_USAGE_GPU_TO_CPU,
		};

		VmaAllocationInfo allocationInfo = {};
		VK_CHECK(vmaCreateBuffer(vmaAllocator, &bufferInfo, &memoryInfo, &texture.cpuReadbackBuffer->handle, &texture.allocation, &allocationInfo));

		VK_CHECK(vmaMapMemory(vmaAllocator, texture.allocation, &texture.cpuReadbackBuffer->data));

		return textureIndex;
	}

	VkImageCreateFlags flags = 0;
	if (desc->type == TextureType::TextureCube)
	{
		flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	}
	VkImageUsageFlags usage = GetVkImageUsageFlags(desc->flags);
	VkImageCreateInfo imageInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.flags = flags,
		.imageType = texture.type,
		.format = texture.format,
		.extent = { texture.width, texture.height, texture.depth },
		.mipLevels = texture.mipLevels,
		.arrayLayers = texture.arrayLayers,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE
	};

	VmaAllocationCreateInfo memoryInfo = {};
	memoryInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	if (imageInfo.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)
	{
		memoryInfo.preferredFlags = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
	}
	VK_CHECK(vmaCreateImage(vmaAllocator, &imageInfo, &memoryInfo, &texture.handle, &texture.allocation, VULKAN_ALLOCATION_CALLBACKS));
	
	if (name)
	{
		SetDebugUtilsObjectName(VK_OBJECT_TYPE_IMAGE, (uint64)texture.handle, name);
	}

	if (HAS_ANY_FLAGS(desc->flags, TextureCreateFlags::ShaderResource))
	{
		VkImageViewCreateInfo imageViewInfo = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = texture.handle,
			.viewType = ConvertToVkImageViewType(desc->type, texture.arrayLayers > 1 ? true : false),
			.format = texture.format,
			.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A },
			.subresourceRange = { texture.aspectMask, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS }
		};
		VK_CHECK(vkCreateImageView(handle, &imageViewInfo, VULKAN_ALLOCATION_CALLBACKS, &texture.srv));

		uint32 index = bindlessManager.AllocateSampledImageIndex();
		VkDescriptorImageInfo descriptorImageInfo = {
			.imageView = texture.srv,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};
		VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = bindlessManager.set,
			.dstBinding = VULKAN_RENDER_BACKEND_BINDLESS_DESCRIPTOR_SLOT_SAMPLED_IMAGES,
			.dstArrayElement = index,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
			.pImageInfo = &descriptorImageInfo,
		};
		vkUpdateDescriptorSets(handle, 1, &write, 0, nullptr);
		texture.srvIndex = index;
	}
	if (HAS_ANY_FLAGS(desc->flags, TextureCreateFlags::UnorderedAccess))
	{
		texture.uavs.resize(texture.mipLevels);
		for (uint32 mipLevel = 0; mipLevel < texture.mipLevels; mipLevel++)
		{
			VkImageViewCreateInfo imageViewInfo = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = texture.handle,
				.viewType = ConvertToVkImageViewType(desc->type, false),
				.format = texture.format,
				.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A },
				.subresourceRange = { texture.aspectMask, mipLevel, 1, 0, VK_REMAINING_ARRAY_LAYERS }
			};
			VK_CHECK(vkCreateImageView(handle, &imageViewInfo, VULKAN_ALLOCATION_CALLBACKS, &texture.uavs[mipLevel].uav));

			uint32 index = bindlessManager.AllocateStorageImageIndex();
			VkDescriptorImageInfo descriptorImageInfo = {
				.imageView = texture.uavs[mipLevel].uav,
				.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
			};
			VkWriteDescriptorSet write = {
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = bindlessManager.set,
				.dstBinding = VULKAN_RENDER_BACKEND_BINDLESS_DESCRIPTOR_SLOT_STORAGE_IMAGES,
				.dstArrayElement = index,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.pImageInfo = &descriptorImageInfo,
			};
			vkUpdateDescriptorSets(handle, 1, &write, 0, nullptr);
			texture.uavs[mipLevel].uavIndex = index;
		}
	}
	if (HAS_ANY_FLAGS(desc->flags, TextureCreateFlags::RenderTarget))
	{
		VkImageViewCreateInfo imageViewInfo = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = texture.handle,
			.viewType = ConvertToVkImageViewType(desc->type, false),
			.format = texture.format,
			.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A },
			.subresourceRange = { texture.aspectMask, 0, 1, 0, 1 }
		};
		VK_CHECK(vkCreateImageView(handle, &imageViewInfo, VULKAN_ALLOCATION_CALLBACKS, &texture.rtv));
	}
	if (HAS_ANY_FLAGS(desc->flags, TextureCreateFlags::DepthStencil))
	{
		for (uint32 i = 0; i < texture.arrayLayers; i++)
		{
			VkImageView dsv = VK_NULL_HANDLE;
			VkImageViewCreateInfo imageViewInfo = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = texture.handle,
				.viewType = ConvertToVkImageViewType(desc->type, false),
				.format = texture.format,
				.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A },
				.subresourceRange = { texture.aspectMask, 0, 1, i, 1 }
			};
			VK_CHECK(vkCreateImageView(handle, &imageViewInfo, VULKAN_ALLOCATION_CALLBACKS, &dsv));
			texture.dsv.push_back(dsv);
		}
	}

	if (data != nullptr)
	{
		RenderBackendBufferDesc uploadBufferDesc = RenderBackendBufferDesc::Create(1, (uint32)texture.allocation->GetSize(), BufferCreateFlags::CpuOnly | BufferCreateFlags::CreateMapped);
		uint32 bufferIndex = CreateBuffer(&uploadBufferDesc, "UploadBuffer");
		VulkanBuffer& uploadBuffer = buffers[bufferIndex];
		MapBuffer(bufferIndex);
		std::vector<VkBufferImageCopy> copyRegions;
		VkDeviceSize copyOffset = 0;
		uint64 dataOffset = 0;
		for (uint32 layer = 0; layer < desc->arrayLayers; layer++)
		{
			uint32 width = imageInfo.extent.width;
			uint32 height = imageInfo.extent.height;
			uint32 depth = imageInfo.extent.depth;
			for (uint32 level = 0; level < desc->mipLevels; level++)
			{
				uint64 copySize = width * height * depth * GetPixelFormatBytes(desc->format);
				uint8* copyDst = (uint8*)uploadBuffer.mappedData + copyOffset;
				memcpy(copyDst, (uint8*)data + dataOffset, copySize);

				VkBufferImageCopy copyRegion = {};
				copyRegion.bufferOffset = copyOffset;
				copyRegion.bufferRowLength = 0;
				copyRegion.bufferImageHeight = 0;
				copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				copyRegion.imageSubresource.mipLevel = level;
				copyRegion.imageSubresource.baseArrayLayer = layer;
				copyRegion.imageSubresource.layerCount = 1;
				copyRegion.imageOffset = { 0, 0, 0 };
				copyRegion.imageExtent = { width, height, depth };

				copyRegions.emplace_back(copyRegion);
				copyOffset += copySize;

				width = std::max(1u, width / 2);
				height = std::max(1u, height / 2);
				depth = std::max(1u, depth / 2);
			}
		}
		UnmapBuffer(bufferIndex);

		VkCommandBuffer commandBuffer; VkCommandPool pool;
		VulkanHelper::CreateTemporaryCommandBuffer(handle, GetQueueFamilyIndex(QueueFamily::Graphics), pool, commandBuffer);
		
		{
			VkImageMemoryBarrier barrier = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				.srcAccessMask = 0,
				.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = texture.handle,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = imageInfo.mipLevels,
					.baseArrayLayer = 0,
					.layerCount = imageInfo.arrayLayers,
				},
			};
		
			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			vkCmdCopyBufferToImage(
				commandBuffer,
				uploadBuffer.handle,
				texture.handle,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				(uint32)copyRegions.size(), 
				copyRegions.data());

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier);
		}

		for (uint32 mipLevel = 1; mipLevel < texture.mipLevels; mipLevel++)
		{
			VkImageBlit imageBlit = {
				.srcSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = mipLevel - 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.srcOffsets = {
					{ .x = 0, .y = 0, .z = 0 },
					{ .x = std::max((int32)(texture.width >> (mipLevel - 1)), 1), .y = std::max((int32)(texture.height >> (mipLevel - 1)), 1), .z = 1,},
				},
				.dstSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = mipLevel,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.dstOffsets = {
					{.x = 0, .y = 0, .z = 0 },
					{.x = std::max((int32)(texture.width >> mipLevel), 1), .y = std::max((int32)(texture.height >> mipLevel), 1), .z = 1,},
				},
			};

			VkImageMemoryBarrier barrier = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				.srcAccessMask = 0,
				.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = texture.handle,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = mipLevel,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			vkCmdBlitImage(commandBuffer,
				texture.handle,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				texture.handle,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&imageBlit,
				VK_FILTER_LINEAR);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier);
		}

		{
			VkImageMemoryBarrier barrier = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
				.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = texture.handle,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = texture.mipLevels,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier); 
		}

		VulkanHelper::FlushTemporaryCommandBuffer(handle, GetCommandQueue(QueueFamily::Graphics, 0)->handle, pool, commandBuffer);
		DestroyBuffer(bufferIndex);
	}

	return textureIndex;
}

void VulkanDevice::DestroyTexture(uint32 index)
{

}

uint32 VulkanDevice::CreateTextureSRV(uint32 textureIndex, const RenderBackendTextureSRVDesc* desc, const char* name)
{
	VulkanTexture& texture = textures[textureIndex];
	VkImageViewCreateInfo imageViewInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = texture.handle,
		.viewType = ConvertToVkImageViewType(texture.t, desc->numArrayLayers > 1),
		.format = texture.format,
		.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A },
		.subresourceRange = { texture.aspectMask, desc->baseMipLevel, desc->numMipLevels, desc->baseArrayLayer, desc->numArrayLayers }
	};
	VK_CHECK(vkCreateImageView(handle, &imageViewInfo, VULKAN_ALLOCATION_CALLBACKS, &texture.srv));

	uint32 index = bindlessManager.AllocateSampledImageIndex();
	VkDescriptorImageInfo descriptorImageInfo = {
		.imageView = texture.srv,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};
	VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = bindlessManager.set,
		.dstBinding = VULKAN_RENDER_BACKEND_BINDLESS_DESCRIPTOR_SLOT_SAMPLED_IMAGES,
		.dstArrayElement = index,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
		.pImageInfo = &descriptorImageInfo,
	};
	vkUpdateDescriptorSets(handle, 1, &write, 0, nullptr);
	texture.srvIndex = index;
	return index;
}

int32 VulkanDevice::GetTextureSRVDescriptorIndex(uint32 textureIndex)
{
	return textures[textureIndex].srvIndex;
}

uint32 VulkanDevice::CreateTextureUAV(uint32 textureIndex, const RenderBackendTextureUAVDesc* desc, const char* name)
{
	VulkanTexture& texture = textures[textureIndex];
	VkImageViewCreateInfo imageViewInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = texture.handle,
		.viewType = ConvertToVkImageViewType(texture.t, false),
		.format = texture.format,
		.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A },
		.subresourceRange = { texture.aspectMask, desc->mipLevel, 1, 0, 1 }
	};
	VK_CHECK(vkCreateImageView(handle, &imageViewInfo, VULKAN_ALLOCATION_CALLBACKS, &texture.uavs[desc->mipLevel].uav));

	uint32 index = bindlessManager.AllocateSampledImageIndex();
	VkDescriptorImageInfo descriptorImageInfo = {
		.imageView = texture.uavs[desc->mipLevel].uav,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};
	VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = bindlessManager.set,
		.dstBinding = VULKAN_RENDER_BACKEND_BINDLESS_DESCRIPTOR_SLOT_SAMPLED_IMAGES,
		.dstArrayElement = index,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
		.pImageInfo = &descriptorImageInfo,
	};
	vkUpdateDescriptorSets(handle, 1, &write, 0, nullptr);
	texture.uavs[desc->mipLevel].uavIndex = index;
	return index;
}

int32 VulkanDevice::GetTextureUAVDescriptorIndex(uint32 textureIndex, uint32 mipLevel)
{
	VulkanTexture& texture = textures[textureIndex];
	return texture.uavs[mipLevel].uavIndex;
}

uint32 VulkanDevice::CreateSampler(const RenderBackendSamplerDesc* desc, const char* name)
{
	VulkanSampler sampler = {};

	VkFilter minFilter, magFilter;
	VkSamplerMipmapMode mipmapMode;
	bool anisotropyEnable;
	bool compareEnable;
	GetVkFilterAndVkSamplerMipmapMode(desc->filter, &minFilter, &magFilter, &mipmapMode, &anisotropyEnable, &compareEnable);

	void* next = nullptr;
	VkSamplerReductionModeCreateInfo reductionModeInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO 
	};

	VkCompareOp compareOp = ConvertToVkCompareOp(desc->compareOp);
	if (false)
	{
		switch (desc->filter)
		{
		case Filter::MinimumMinMagMipPoint:
		case Filter::MinimumMinMagPointMipLinear:
		case Filter::MinimumMinPointMagLinearMipPoint:
		case Filter::MinimumMinPointMagMipLinear:
		case Filter::MinimumMinLinearMagMipPoint:
		case Filter::MinimumMinLinearMagPointMipLinear:
		case Filter::MinimumMinMagLinearMipPoint:
		case Filter::MinimumMinMagMipLinear:
		case Filter::MinimumAnisotropic:
			reductionModeInfo.reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN;
			next = &reductionModeInfo;
			break;
		case Filter::MaximumMinMagMipPoint:
		case Filter::MaximumMinMagPointMipLinear:
		case Filter::MaximumMinPointMagLinearMipPoint:
		case Filter::MaximumMinPointMagMipLinear:
		case Filter::MaximumMinLinearMagMipPoint:
		case Filter::MaximumMinLinearMagPointMipLinear:
		case Filter::MaximumMinMagLinearMipPoint:
		case Filter::MaximumMinMagMipLinear:
		case Filter::MaximumAnisotropic:
			reductionModeInfo.reductionMode = VK_SAMPLER_REDUCTION_MODE_MAX;
			next = &reductionModeInfo;
			break;
		default:
			break;
		}
	}

	VkSamplerCreateInfo samplerInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.pNext = next,
		.magFilter = minFilter,
		.minFilter = magFilter,
		.mipmapMode = mipmapMode,
		.addressModeU = ConvertToVkSamplerAddressMode(desc->addressModeU),
		.addressModeV = ConvertToVkSamplerAddressMode(desc->addressModeV),
		.addressModeW = ConvertToVkSamplerAddressMode(desc->addressModeW),
		.mipLodBias = desc->mipLodBias,
		.anisotropyEnable = ConvertToVkBool(anisotropyEnable),
		.maxAnisotropy = (float)desc->maxAnisotropy,
		.compareEnable = ConvertToVkBool(compareEnable),
		.compareOp = compareEnable ? compareOp : VK_COMPARE_OP_NEVER,
		.minLod = desc->minLod,
		.maxLod = desc->maxLod,
		.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
		.unnormalizedCoordinates = VK_FALSE,
	};

	VK_CHECK(vkCreateSampler(handle, &samplerInfo, VULKAN_ALLOCATION_CALLBACKS, &sampler.handle));

	sampler.bindlessIndex = bindlessManager.AllocateSamplerIndex();

	VkDescriptorImageInfo imageInfo = { 
		.sampler = sampler.handle
	};
	VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = bindlessManager.set,
		.dstBinding = BindlessBindingSamplers,
		.dstArrayElement = sampler.bindlessIndex,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
		.pImageInfo = &imageInfo,
	};
	vkUpdateDescriptorSets(handle, 1, &write, 0, nullptr);

	uint32 samplerIndex = 0;
	if (!freeSamplers.empty())
	{
		samplerIndex = freeSamplers.back();
		freeSamplers.pop_back();
		samplers[samplerIndex] = sampler;
	}
	else
	{
		samplerIndex = (uint32)samplers.size();
		samplers.emplace_back(sampler);
	}
	return samplerIndex;
}

void VulkanDevice::DestroySampler(uint32 index)
{

}

static void InitRasterizationStateInfo(const RasterizationState& state, VkPipelineRasterizationStateCreateInfo& outInfo)
{
	outInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	outInfo.depthClampEnable = state.depthClampEnable;
	outInfo.depthBiasEnable = VK_FALSE;
	outInfo.depthBiasConstantFactor = state.depthBiasConstantFactor;
	outInfo.depthBiasClamp = 1.0f;
	outInfo.depthBiasSlopeFactor = state.depthBiasSlopeFactor;
	outInfo.rasterizerDiscardEnable = VK_FALSE;
	outInfo.polygonMode = state.fillMode == RasterizationFillMode::Solid ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_LINE;
	outInfo.cullMode = ConvertToVkCullModeFlags(state.cullMode);
	outInfo.frontFace = state.frontFaceCounterClockwise ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
	outInfo.lineWidth = 1.0f;
}

static void InitDepthStencilStateInfo(const DepthStencilState& state, VkPipelineDepthStencilStateCreateInfo& outInfo)
{
	outInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	outInfo.depthTestEnable = ConvertToVkBool(state.depthTestEnable);
	outInfo.depthWriteEnable = ConvertToVkBool(state.depthWriteEnable);
	outInfo.depthBoundsTestEnable = VK_FALSE;
	outInfo.depthCompareOp = ConvertToVkCompareOp(state.depthCompareOp);
	outInfo.stencilTestEnable = ConvertToVkBool(state.stencilTestEnable);
	outInfo.front = ConvertToVkStencilOpState(state.front);
	outInfo.back = ConvertToVkStencilOpState(state.back);
}

static void InitColorBlendStateInfo(const ColorBlendState& state, VkPipelineColorBlendAttachmentState* attachmentStates, VkPipelineColorBlendStateCreateInfo& outInfo)
{
	uint32 attachmentCount = state.numColorAttachments;
	for (uint32 i = 0; i < attachmentCount; i++)
	{
		attachmentStates[i].blendEnable = ConvertToVkBool(state.attachmentStates[i].blendEnable);
		attachmentStates[i].srcColorBlendFactor = (VkBlendFactor)state.attachmentStates[i].srcColorBlendFactor;
		attachmentStates[i].dstColorBlendFactor = (VkBlendFactor)state.attachmentStates[i].dstColorBlendFactor;
		attachmentStates[i].colorBlendOp = (VkBlendOp)state.attachmentStates[i].colorBlendOp;
		attachmentStates[i].srcAlphaBlendFactor = (VkBlendFactor)state.attachmentStates[i].srcAlphaBlendFactor;
		attachmentStates[i].dstAlphaBlendFactor = (VkBlendFactor)state.attachmentStates[i].dstAlphaBlendFactor;
		attachmentStates[i].alphaBlendOp = (VkBlendOp)state.attachmentStates[i].alphaBlendOp;
		attachmentStates[i].colorWriteMask = (VkColorComponentFlags)state.attachmentStates[i].colorWriteMask;
	}

	outInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	outInfo.logicOpEnable = VK_FALSE;
	outInfo.logicOp = VK_LOGIC_OP_COPY;
	outInfo.attachmentCount = attachmentCount;
	//outInfo.pAttachments = attachmentStates;
	outInfo.blendConstants[0] = state.blendFactor[0];
	outInfo.blendConstants[1] = state.blendFactor[1];
	outInfo.blendConstants[2] = state.blendFactor[2];
	outInfo.blendConstants[3] = state.blendFactor[3];
}

uint32 VulkanDevice::CreateShader(const RenderBackendShaderDesc* desc, const char* name)
{
	uint32 shaderIndex = 0;
	if (!freeShaders.empty())
	{
		shaderIndex = freeShaders.back();
		freeShaders.pop_back();
	}
	else
	{
		shaderIndex = (uint32)shaders.size();
		shaders.emplace_back();
	}
	VulkanShader& shader = shaders[shaderIndex];
	
	for (uint32 stageIndex = 0; stageIndex < (uint32)RenderBackendShaderStage::Count; stageIndex++)
	{
		RenderBackendShaderStage stage = (RenderBackendShaderStage)stageIndex;
		if (desc->stages[stageIndex].size == 0)
		{
			continue;
		}

		shader.entryPoints[stageIndex] = desc->entryPoints[stageIndex].c_str();
		VkPipelineShaderStageCreateInfo& shaderStageInfo = shader.stages[shader.numStages];
		shaderStageInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = ConvertToVkShaderStageFlagBits(stage),
			.pName = shader.entryPoints[stageIndex].c_str(),
		};
		VkShaderModuleCreateInfo shaderModuleInfo = {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = desc->stages[stageIndex].size,
			.pCode = (uint32*)desc->stages[stageIndex].data,
		};
		VK_CHECK(vkCreateShaderModule(handle, &shaderModuleInfo, VULKAN_ALLOCATION_CALLBACKS, &shaderStageInfo.module));
		SetDebugUtilsObjectName(VK_OBJECT_TYPE_SHADER_MODULE, (uint64)shaderStageInfo.module, name);
		shader.numStages++; 
	}		

	InitRasterizationStateInfo(desc->rasterizationState, shader.rasterizationState);
	InitDepthStencilStateInfo(desc->depthStencilState, shader.depthStencilState);
	InitColorBlendStateInfo(desc->colorBlendState, shader.colorBlendAttachmentStates, shader.colorBlendState);

	return shaderIndex;
}

void VulkanDevice::DestroyShader(uint32 index)
{

}

uint32 VulkanDevice::CreateAccelerationStructure(VulkanRayTracingAccelerationStructure* accelerationStructure, VkAccelerationStructureTypeKHR type, uint32* primitiveCounts, const char* name)
{
	VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.type = type,
		.flags = accelerationStructure->buildFlags,
		.geometryCount = (uint32)accelerationStructure->geometries.size(),
		.pGeometries = accelerationStructure->geometries.data(),
	};

	VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR 
	};
	backend->functions.vkGetAccelerationStructureBuildSizesKHR(
		handle,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&accelerationStructureBuildGeometryInfo,
		primitiveCounts,
		&accelerationStructureBuildSizesInfo);

	// Create Acceleration Structure Buffer
	{
		accelerationStructure->accelerationStructureBuffer.size = accelerationStructureBuildSizesInfo.accelerationStructureSize;
		VkBufferCreateInfo bufferCreateInfo = { 
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = accelerationStructure->accelerationStructureBuffer.size,
			.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		}; 
		VmaAllocationCreateInfo memoryInfo = {
			.flags = 0,
			.usage = VMA_MEMORY_USAGE_GPU_ONLY,
		};
		VmaAllocationInfo allocationInfo = {};
		VK_CHECK(vmaCreateBuffer(
			vmaAllocator,
			&bufferCreateInfo,
			&memoryInfo,
			&accelerationStructure->accelerationStructureBuffer.buffer, 
			&accelerationStructure->accelerationStructureBuffer.allocation,
			&accelerationStructure->accelerationStructureBuffer.allocationInfo));
		SetDebugUtilsObjectName(VK_OBJECT_TYPE_BUFFER, (uint64)accelerationStructure->accelerationStructureBuffer.buffer, "Acceleration Structure Buffer");
		VkBufferDeviceAddressInfoKHR bufferDeviceInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = accelerationStructure->accelerationStructureBuffer.buffer,
		};
		accelerationStructure->accelerationStructureBuffer.deviceAddress = backend->functions.vkGetBufferDeviceAddressKHR(handle, &bufferDeviceInfo);
	}

	// Create Scratch Buffer
	{
		accelerationStructure->scratchBuffer.size = accelerationStructureBuildSizesInfo.buildScratchSize;
		VkBufferCreateInfo bufferCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = accelerationStructure->scratchBuffer.size,
			.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		};
		VmaAllocationCreateInfo memoryInfo = {
			.flags = 0,
			.usage = VMA_MEMORY_USAGE_GPU_ONLY,
		};
		VmaAllocationInfo allocationInfo = {};
		VK_CHECK(vmaCreateBuffer(
			vmaAllocator,
			&bufferCreateInfo,
			&memoryInfo,
			&accelerationStructure->scratchBuffer.buffer,
			&accelerationStructure->scratchBuffer.allocation,
			&accelerationStructure->scratchBuffer.allocationInfo));
		SetDebugUtilsObjectName(VK_OBJECT_TYPE_BUFFER, (uint64)accelerationStructure->scratchBuffer.buffer, "Acceleration Structure Scratch Buffer");
		VkBufferDeviceAddressInfoKHR bufferDeviceInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = accelerationStructure->scratchBuffer.buffer,
		};
		accelerationStructure->scratchBuffer.deviceAddress = backend->functions.vkGetBufferDeviceAddressKHR(handle, &bufferDeviceInfo);
	}

	VkAccelerationStructureCreateInfoKHR accelerationStructureInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
		.buffer = accelerationStructure->accelerationStructureBuffer.buffer,
		.size = accelerationStructureBuildSizesInfo.accelerationStructureSize,
		.type = type
	};
	VK_CHECK(backend->functions.vkCreateAccelerationStructureKHR(
		handle, 
		&accelerationStructureInfo,
		VULKAN_ALLOCATION_CALLBACKS, 
		&accelerationStructure->handle));
	SetDebugUtilsObjectName(VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, (uint64)accelerationStructure->handle, name);

	VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
		.accelerationStructure = accelerationStructure->handle,
	};
	accelerationStructure->deviceAddress = backend->functions.vkGetAccelerationStructureDeviceAddressKHR(handle, &deviceAddressInfo);
	
	VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.type = type,
		.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
		.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		.dstAccelerationStructure = accelerationStructure->handle,
		.geometryCount = (uint32)accelerationStructure->geometries.size(),
		.pGeometries = accelerationStructure->geometries.data(),
		.scratchData = { .deviceAddress = accelerationStructure->scratchBuffer.deviceAddress }
	};

	std::vector<VkAccelerationStructureBuildRangeInfoKHR> accelerationBuildStructureRangeInfos((uint32)accelerationStructure->geometries.size());
	std::vector<VkAccelerationStructureBuildRangeInfoKHR*> pBuildRangeInfos((uint32)accelerationStructure->geometries.size());
	for (uint32 geometryIndex = 0; geometryIndex < (uint32)accelerationStructure->geometries.size(); geometryIndex++)
	{
		accelerationBuildStructureRangeInfos[geometryIndex] = {
			.primitiveCount = primitiveCounts[geometryIndex],
			.primitiveOffset = 0,
			.firstVertex = 0,
			.transformOffset = 0,
		};
		pBuildRangeInfos[geometryIndex] = &accelerationBuildStructureRangeInfos[geometryIndex];
	}

	VkCommandBuffer commandBuffer; VkCommandPool pool;
	VulkanHelper::CreateTemporaryCommandBuffer(handle, GetQueueFamilyIndex(QueueFamily::Graphics), pool, commandBuffer);
	backend->functions.vkCmdBuildAccelerationStructuresKHR(
		commandBuffer,
		1,
		&accelerationBuildGeometryInfo, 
		pBuildRangeInfos.data());
	VulkanHelper::FlushTemporaryCommandBuffer(handle, GetCommandQueue(QueueFamily::Graphics, 0)->handle, pool, commandBuffer);

	if (type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR || type == VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR)
	{
		uint32 descriptorIndex = bindlessManager.AllocateAccelerationStructureIndex();
		const VkWriteDescriptorSetAccelerationStructureKHR writeAccelerationStructureInfo = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
			.accelerationStructureCount = 1,
			.pAccelerationStructures = &accelerationStructure->handle,
		};
		VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.pNext = &writeAccelerationStructureInfo,
			.dstSet = bindlessManager.set,
			.dstBinding = VULKAN_RENDER_BACKEND_BINDLESS_DESCRIPTOR_SLOT_ACCELERATION_STRUCTURES,
			.dstArrayElement = descriptorIndex,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
		};
		vkUpdateDescriptorSets(handle, 1, &write, 0, nullptr);
		accelerationStructure->descriptorIndex = descriptorIndex;
	}

	uint32 index = 0;
	if (!freeAccelerationStructures.empty())
	{
		index = freeAccelerationStructures.back();
		freeAccelerationStructures.pop_back();
		accelerationStructures[index] = *accelerationStructure;
	}
	else
	{
		index = (uint32)accelerationStructures.size();
		accelerationStructures.emplace_back(*accelerationStructure);
	}
	return index;
}

uint32 VulkanDevice::CreateBottomLevelAS(const RenderBackendBottomLevelASDesc* desc, const char* name)
{
	VulkanRayTracingAccelerationStructure accelerationStructure;
	accelerationStructure.buildFlags = ConvertToVkBuildAccelerationStructureFlagsKHR(desc->buildFlags);
	accelerationStructure.blasDesc = *desc;

	std::vector<uint32> primitiveCounts(desc->numGeometries);
	for (uint32 i = 0; i < desc->numGeometries; i++)
	{
		const RenderBackendGeometryDesc& geometryDesc = desc->geometryDescs[i];
		VkAccelerationStructureGeometryKHR geometry = {
		   .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		   .geometryType = (VkGeometryTypeKHR)geometryDesc.type,
		   .flags = ConvertToVkGeometryFlagsKHR(geometryDesc.flags),
		};
		if (geometry.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR)
		{
			uint32 maxVertex = geometryDesc.triangleDesc.numVertices;
			geometry.geometry.triangles = {
				.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
				.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
				.vertexData = GetBufferDeviceAddress(geometryDesc.triangleDesc.vertexBuffer) + geometryDesc.triangleDesc.vertexOffset,
				.vertexStride = geometryDesc.triangleDesc.vertexStride,
				.maxVertex = maxVertex,
				.indexType = VK_INDEX_TYPE_UINT32,
				.indexData = GetBufferDeviceAddress(geometryDesc.triangleDesc.indexBuffer) + geometryDesc.triangleDesc.indexOffset,
				.transformData = GetBufferDeviceAddress(geometryDesc.triangleDesc.transformBuffer) + geometryDesc.triangleDesc.transformOffset
			};
			primitiveCounts[i] = geometryDesc.triangleDesc.numIndices / 3;
		}
		else if (geometry.geometryType == VK_GEOMETRY_TYPE_AABBS_KHR)
		{
			geometry.geometry.aabbs = {
			   .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
			   .data = GetBufferDeviceAddress(geometryDesc.aabbDesc.buffer) + geometryDesc.aabbDesc.offset,
			   .stride = sizeof(VkAabbPositionsKHR)
			}; 
			VulkanBuffer* buffer = GetBuffer(geometryDesc.aabbDesc.buffer); 
			primitiveCounts[i] = (uint32)(buffer->size / sizeof(VkAabbPositionsKHR));
		}
		accelerationStructure.geometries.emplace_back(geometry);
	}
	uint32 index = CreateAccelerationStructure(&accelerationStructure, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, primitiveCounts.data(), name);

	return index;
}

uint32 VulkanDevice::CreateTopLevelAS(const RenderBackendTopLevelASDesc* desc, const char* name)
{
	VulkanRayTracingAccelerationStructure accelerationStructure = {};
	accelerationStructure.buildFlags = ConvertToVkBuildAccelerationStructureFlagsKHR(desc->buildFlags);
	accelerationStructure.tlasDesc = *desc;

	std::vector<VkAccelerationStructureInstanceKHR> instances;
	for (uint32 i = 0; i < desc->numInstances; i++)
	{
		VkTransformMatrixKHR transformMatrix;
	    memcpy(&transformMatrix, &desc->instances[i].transformMatrix, sizeof(VkTransformMatrixKHR));
		instances.emplace_back(VkAccelerationStructureInstanceKHR{
			.transform = transformMatrix,
			.instanceCustomIndex = desc->instances[i].instanceID,
			.mask = desc->instances[i].instanceMask,
			.instanceShaderBindingTableRecordOffset = desc->instances[i].instanceContributionToHitGroupIndex,
			.flags = ConvertToVkGeometryInstanceFlagsKHR(desc->instances[i].flags),
			.accelerationStructureReference = GetAccelerationStructure(desc->instances[i].blas)->deviceAddress,
		});
	}

	VulkanRayTracingAccelerationStructure::Buffer instanceBuffer;
	{
		instanceBuffer.size = desc->numInstances * sizeof(VkAccelerationStructureInstanceKHR);
		VkBufferCreateInfo bufferCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = instanceBuffer.size,
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		};
		VmaAllocationCreateInfo memoryInfo = {
			.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
			.usage = VMA_MEMORY_USAGE_CPU_ONLY,
		};
		VmaAllocationInfo allocationInfo = {};
		VK_CHECK(vmaCreateBuffer(
			vmaAllocator,
			&bufferCreateInfo,
			&memoryInfo,
			&instanceBuffer.buffer,
			&instanceBuffer.allocation,
			&instanceBuffer.allocationInfo));
		SetDebugUtilsObjectName(VK_OBJECT_TYPE_BUFFER, (uint64)instanceBuffer.buffer, "Acceleration Structure Instance Buffer");
		VkBufferDeviceAddressInfoKHR bufferDeviceInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = instanceBuffer.buffer,
		};
		instanceBuffer.deviceAddress = backend->functions.vkGetBufferDeviceAddressKHR(handle, &bufferDeviceInfo);
		memcpy(instanceBuffer.allocationInfo.pMappedData, instances.data(), instanceBuffer.size);
		VK_CHECK(vmaFlushAllocation(vmaAllocator, instanceBuffer.allocation, 0, instanceBuffer.size));
		accelerationStructure.resourceBuffers.emplace_back(instanceBuffer);
	}

	VkAccelerationStructureGeometryKHR geometry = {
	    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
	    .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
		.geometry = { 
			.instances = {
				.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
				.arrayOfPointers = VK_FALSE,
				.data = {
					.deviceAddress = instanceBuffer.deviceAddress
                },
			}, 
		},
		.flags = ConvertToVkGeometryFlagsKHR(desc->geometryFlags),
	};
	accelerationStructure.geometries.emplace_back(geometry);

	uint32 numPrimitives = desc->numInstances;
	uint32 index = CreateAccelerationStructure(&accelerationStructure, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &numPrimitives, name);

	return index;
}

void VulkanDevice::ResizeSwapChain(uint32 index, uint32* width, uint32* height)
{
	VulkanSwapchain* swapchain = &swapchains[index];
	if (swapchain->info.imageExtent.width == *width && swapchain->info.imageExtent.height == *height)
	{
		return;
	}
	RecreateSwapChain(index);
	*width = swapchain->info.imageExtent.width;
	*height = swapchain->info.imageExtent.height;
}

VulkanSwapchain::Status VulkanDevice::AcquireImageIndex(uint32 index)
{
	VulkanSwapchain* swapchain = &swapchains[index];
	uint32 semaphoreIndex = (swapchain->semaphoreIndex + 1) % swapchain->numBuffers;

	VkSemaphore& imageAcquiredSemaphore = swapchain->imageAcquiredSemaphores[semaphoreIndex];
	VkFence& imageAcquiredFence = swapchain->imageAcquiredFences[semaphoreIndex];

	// TODO: remove this
	vkDeviceWaitIdle(handle);

	VK_CHECK(vkWaitForFences(handle, 1, &imageAcquiredFence, VK_TRUE, UINT64_MAX));
	VK_CHECK(vkResetFences(handle, 1, &imageAcquiredFence));

	uint32 imageIndex = 0;
	VkResult result = vkAcquireNextImageKHR(handle, swapchain->handle, UINT64_MAX, imageAcquiredSemaphore, imageAcquiredFence, &imageIndex);
	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		return VulkanSwapchain::Status::OutOfDate;
	}
	ASSERT(result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR);
	swapchain->semaphoreIndex = semaphoreIndex;
	swapchain->activeBackBufferIndex = imageIndex;
	return VulkanSwapchain::Status::Success;
}

VulkanSwapchain::Status VulkanDevice::PresentSwapChain(uint32 index, VkSemaphore* waitSemaphores, uint32 waitSemaphoreCount)
{
	VulkanSwapchain* swapchain = &swapchains[index];
	VkPresentInfoKHR presentInfo = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = waitSemaphoreCount,
		.pWaitSemaphores = waitSemaphores,
		.swapchainCount = 1,
		.pSwapchains = &swapchain->handle,
		.pImageIndices = &swapchain->activeBackBufferIndex,
	};
	VkQueue presentQueue = commandQueues[(uint32)QueueFamily::Graphics][0].handle;
	VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);
	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		return VulkanSwapchain::Status::OutOfDate;
	}
	ASSERT(result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR);
	return VulkanSwapchain::Status::Success;
}

VkRenderPass VulkanDevice::FindOrCreateRenderPass(const VulkanRenderPassDesc& renderPassDesc)
{
	VkRenderPass renderPass = VK_NULL_HANDLE;

	uint32 renderPassHash = renderPassDesc.renderPassFullHash;
	if (cachedRenderPasses.find(renderPassHash) != cachedRenderPasses.end())
	{
		renderPass = cachedRenderPasses[renderPassHash];
	}
	else
	{
		std::array<VkSubpassDependency, 2> subpassDependencies;
		subpassDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		subpassDependencies[0].dstSubpass = 0;
		subpassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		subpassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		subpassDependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		subpassDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		subpassDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		subpassDependencies[1].srcSubpass = 0;
		subpassDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		subpassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		subpassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		subpassDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		subpassDependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		subpassDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkSubpassDescription subpassDesc = {
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = renderPassDesc.numColorAttachments,
			.pColorAttachments = renderPassDesc.colorReferences,
			.pDepthStencilAttachment = renderPassDesc.hasDepthStencil ? &renderPassDesc.depthStencilReference : nullptr,
		};
		VkRenderPassCreateInfo renderPassInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = renderPassDesc.numAttachmentDescriptions,
			.pAttachments = renderPassDesc.attachmentDescriptions,
			.subpassCount = 1,
			.pSubpasses = &subpassDesc,
			.dependencyCount = (uint32)subpassDependencies.size(),
			.pDependencies = subpassDependencies.data(),
		};
		VK_CHECK(vkCreateRenderPass(handle, &renderPassInfo, VULKAN_ALLOCATION_CALLBACKS, &renderPass));
		cachedRenderPasses[renderPassHash] = renderPass;
	}

	return renderPass;
}

bool Matches(VulkanDevice* device, const VulkanFramebuffer& framebuffer, const RenderPassInfo& renderPassInfo, uint32 numColorAttachments)
{
	if (framebuffer.numColorAttachments != numColorAttachments)
	{
		return false;
	}

	for (uint32 index = 0; index < framebuffer.numColorAttachments; index++)
	{
		VkImage image1 = framebuffer.images[index];
		VkImage image2 = device->GetTexture(renderPassInfo.colorRenderTargets[index].texture)->handle;
		if (image1 != image2)
		{
			return false;
		}
	}

	if ((framebuffer.numAttachments != framebuffer.numColorAttachments) && renderPassInfo.depthStencilRenderTarget.texture)
	{
		VkImage image1 = framebuffer.images[framebuffer.numColorAttachments];
		VkImage image2 = device->GetTexture(renderPassInfo.depthStencilRenderTarget.texture)->handle;
		if (image1 != image2)
		{
			return false;
		}
	}

	return true;
}

VulkanFramebuffer* VulkanDevice::FindOrCreateFramebuffer(const RenderPassInfo& renderPassInfo, const VulkanRenderPassDesc& renderPassDesc, VkRenderPass renderPass)
{
	uint32 renderPassCompatibleHash = renderPassDesc.renderPassCompatibleHash;

	uint64 mipLevelsAndArrayLayers[MaxNumSimultaneousColorRenderTargets + 1];
	for (int32 index = 0; index < MaxNumSimultaneousColorRenderTargets; index++)
	{
		mipLevelsAndArrayLayers[index] = ((uint64)renderPassInfo.colorRenderTargets[index].arrayLayer << (uint64)32) | (uint64)renderPassInfo.colorRenderTargets[index].mipLevel;
	}
	mipLevelsAndArrayLayers[MaxNumSimultaneousColorRenderTargets] = ((uint64)renderPassInfo.depthStencilRenderTarget.arrayLayer << (uint64)32) | (uint64)renderPassInfo.depthStencilRenderTarget.mipLevel;
	uint32 framebufferHash = Crc32(mipLevelsAndArrayLayers, (MaxNumSimultaneousColorRenderTargets + 1) * sizeof(uint64), renderPassCompatibleHash);

	FramebufferList* framebufferList = nullptr;
	if (cachedFramebuffers.find(framebufferHash) != cachedFramebuffers.end())
	{
		framebufferList = &cachedFramebuffers[framebufferHash];
		for (uint64 index = 0; index < framebufferList->framebuffers.size(); index++)
		{
			if (Matches(this, framebufferList->framebuffers[index], renderPassInfo, renderPassDesc.numColorAttachments))
			{
				return &framebufferList->framebuffers[index];
			}
		}
	}
	else
	{
		cachedFramebuffers.emplace(framebufferHash, FramebufferList{});
		framebufferList = &cachedFramebuffers[framebufferHash];
	}

	const VkExtent3D& extent = renderPassDesc.extent;
	uint32 width = extent.width;
	uint32 height = extent.height;
	uint32 layers = extent.depth;

	VulkanFramebuffer framebuffer = {
		.width = width,
		.height = height,
		.layers = layers,
	};

	uint32 numAttachments = renderPassDesc.numColorAttachments;

	for (uint32 index = 0; index < numAttachments; index++)
	{
		VulkanTexture* texture = GetTexture(renderPassInfo.colorRenderTargets[index].texture);
		uint32 mipLevel = renderPassInfo.colorRenderTargets[index].mipLevel;
		uint32 arrayLayer = renderPassInfo.colorRenderTargets[index].arrayLayer;

		framebuffer.images[index] = texture->handle;
		framebuffer.attachments[index] = texture->rtv;
	}
	if (renderPassDesc.hasDepthStencil)
	{
		VulkanTexture* texture = GetTexture(renderPassInfo.depthStencilRenderTarget.texture);
		uint32 mipLevel = renderPassInfo.depthStencilRenderTarget.mipLevel;
		uint32 arrayLayer = renderPassInfo.depthStencilRenderTarget.arrayLayer;

		bool hasStencil = IsStencilFormat(texture->format);

		framebuffer.images[numAttachments] = texture->handle;
		framebuffer.attachments[numAttachments] = texture->dsv[arrayLayer];
		numAttachments++;
	}
	framebuffer.numColorAttachments = renderPassDesc.numColorAttachments,
	framebuffer.numAttachments = numAttachments;

	ASSERT(layers != (uint32)-1 && layers != 0);

	VkFramebufferCreateInfo frameBufferInfo = { 
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = renderPass,
		.attachmentCount = numAttachments,
		.pAttachments = framebuffer.attachments,
		.width = width,
		.height = height,
		.layers = layers,
	};
	VK_CHECK(vkCreateFramebuffer(handle, &frameBufferInfo, VULKAN_ALLOCATION_CALLBACKS, &framebuffer.handle));

	framebufferList->framebuffers.emplace_back(framebuffer);

	return &framebufferList->framebuffers.back();
}

VkPipelineLayout VulkanDevice::FindOrCreatePipelineLayout(uint32 pushConstantSize, RenderBackendPipelineType pipelineType)
{	
	uint64 layoutHash = Crc32(&pushConstantSize, sizeof(uint32), (uint32)pipelineType);
	if (pipelineManager.pipelineLayoutMap.find(layoutHash) != pipelineManager.pipelineLayoutMap.end())
	{
		return pipelineManager.pipelineLayoutMap[layoutHash];
	}

	VkShaderStageFlags shaderStageFlags = 0;
	switch (pipelineType)
	{
	case RenderBackendPipelineType::Graphics:
		shaderStageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;
		break;
	case RenderBackendPipelineType::Compute:
		shaderStageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		break;
	case RenderBackendPipelineType::RayTracing:
		shaderStageFlags = VK_SHADER_STAGE_ALL;
		break;
	default:
		INVALID_ENUM_VALUE();
		break;
	}
	VkPushConstantRange pushConstantRange = {
		.stageFlags = shaderStageFlags,
		.offset = 0,
		.size = pushConstantSize
	};
	VkPipelineLayoutCreateInfo layoutInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &bindlessManager.layout,
		.pushConstantRangeCount = pushConstantSize ? 1u : 0u,
		.pPushConstantRanges = pushConstantSize ? &pushConstantRange : nullptr,
	};
	VkPipelineLayout pipelineLayout;
	VK_CHECK(vkCreatePipelineLayout(handle, &layoutInfo, VULKAN_ALLOCATION_CALLBACKS, &pipelineLayout));

	pipelineManager.pipelineLayoutMap.emplace(layoutHash, pipelineLayout);
	pipelineManager.pipelineLayouts.push_back(pipelineLayout);

	return pipelineLayout;
}

VulkanPipeline* VulkanDevice::FindOrCreateComputePipeline(VulkanShader* shader, uint32 pushConstantSize)
{
	uint32 pipelineHash = Crc32(shader, sizeof(PipelineState), pushConstantSize);

	if (pipelineManager.pipelineMap.find(pipelineHash) != pipelineManager.pipelineMap.end())
	{
		return &pipelineManager.pipelineMap[pipelineHash];
	}

	VkPipelineLayout pipelineLayout = FindOrCreatePipelineLayout(pushConstantSize, RenderBackendPipelineType::Compute);

	shader->stages[0].pName = shader->entryPoints[2].c_str();
	VkComputePipelineCreateInfo computePipelineInfo = { 
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = shader->stages[0],
		.layout = pipelineLayout,
	};

	VkPipeline pipeline;
	VK_CHECK(vkCreateComputePipelines(handle, pipelineManager.pipelineCache, 1, &computePipelineInfo, VULKAN_ALLOCATION_CALLBACKS, &pipeline));

	pipelineManager.pipelineMap.emplace(pipelineHash, VulkanPipeline{ pipelineHash, pipeline, pipelineLayout });
	pipelineManager.pipelines.push_back(VulkanPipeline{ pipelineHash, pipeline, pipelineLayout });
	shader->pipelineHashes.push_back(pipelineHash);

	return &pipelineManager.pipelineMap[pipelineHash];
}

VulkanPipeline* VulkanDevice::FindOrCreateGraphicsPipeline(VulkanShader* shader, VkRenderPass renderPass, PrimitiveTopology topology, uint32 pushConstantSize)
{
	uint32 pipelineStateHash = Crc32(shader, sizeof(VulkanShader));
	uint64 values[] = { (uint64)renderPass, (uint64)topology, (uint64)pushConstantSize };
	uint64 pipelineHash = (uint64(Crc32(values, 3 * sizeof(uint64))) << 32);

	if (pipelineManager.pipelineMap.find(pipelineHash) != pipelineManager.pipelineMap.end())
	{
		return &pipelineManager.pipelineMap[pipelineHash];
	}

	VkPipelineLayout pipelineLayout = FindOrCreatePipelineLayout(pushConstantSize, RenderBackendPipelineType::Graphics);

	static VkPipelineViewportStateCreateInfo viewportStateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.pViewports = nullptr,
		.scissorCount = 1,
		.pScissors = nullptr,
	};

	static VkPipelineVertexInputStateCreateInfo vertexInputStateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
	};

	VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamicStateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = ARRAY_SIZE(dynamicStates),
		.pDynamicStates = dynamicStates,
	};

	VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = ConvertToVkPrimitiveTopology(topology),
		.primitiveRestartEnable = (topology == PrimitiveTopology::TriangleStrip || topology == PrimitiveTopology::TriangleFan) ? 1u : 0u,
	};

	VkSampleMask sampleMask = (uint32)-1;
	VkPipelineMultisampleStateCreateInfo multisamplingStateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE,
		.minSampleShading = 0.0f,
		.pSampleMask = &sampleMask,
		.alphaToCoverageEnable = VK_FALSE,
		.alphaToOneEnable = VK_FALSE,
	};

	VkPipelineRasterizationStateCreateInfo rasterizationStateInfo = shader->rasterizationState;
	VkPipelineDepthStencilStateCreateInfo depthStencilStateInfo = shader->depthStencilState;
	VkPipelineColorBlendStateCreateInfo colorBlendStateInfo = shader->colorBlendState;
	colorBlendStateInfo.pAttachments = shader->colorBlendAttachmentStates;

	// TODO:
	for (uint32 i = 0; i < shader->numStages; i++)
	{
		shader->stages[i].pName = shader->entryPoints[i].c_str();
	}

	VkGraphicsPipelineCreateInfo graphicsPipelineInfo = { 
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = shader->numStages,
		.pStages = shader->stages,
		.pVertexInputState = &vertexInputStateInfo,
		.pInputAssemblyState = &inputAssemblyStateInfo,
		.pViewportState = &viewportStateInfo,
		.pRasterizationState = &rasterizationStateInfo,
		.pMultisampleState = &multisamplingStateInfo,
		.pDepthStencilState = &depthStencilStateInfo,
		.pColorBlendState = &colorBlendStateInfo,
		.pDynamicState = &dynamicStateInfo,
		.layout = pipelineLayout,
		.renderPass = renderPass,
		.subpass = 0,
	};

	VkPipeline pipeline;
	VK_CHECK(vkCreateGraphicsPipelines(handle, pipelineManager.pipelineCache, 1, &graphicsPipelineInfo, VULKAN_ALLOCATION_CALLBACKS, &pipeline));

	pipelineManager.pipelineMap.emplace(pipelineHash, VulkanPipeline{ pipelineHash, pipeline, pipelineLayout });
	pipelineManager.pipelines.push_back(VulkanPipeline{ pipelineHash, pipeline, pipelineLayout });
	shader->pipelineHashes.push_back(pipelineHash);

	return &pipelineManager.pipelineMap[pipelineHash];
}

void VulkanDevice::RecreateSwapChain(uint32 index)
{
	vkDeviceWaitIdle(handle);

	VulkanSwapchain& swapchain = swapchains[index];
	for (uint32 i = 0; i < swapchain.numBuffers; i++)
	{
		VulkanTexture* texture = GetTexture(swapchain.buffers[i]);
		texture = {};
		RemoveRenderBackendHandleRepresentation(swapchain.buffers[i].GetIndex());
		vkWaitForFences(handle, 1, &swapchain.imageAcquiredFences[i], VK_TRUE, UINT64_MAX);
		vkDestroyFence(handle, swapchain.imageAcquiredFences[i], VULKAN_ALLOCATION_CALLBACKS);
		vkDestroySemaphore(handle, swapchain.imageAcquiredSemaphores[i], VULKAN_ALLOCATION_CALLBACKS);
	}
	VkSurfaceCapabilitiesKHR surfaceCapabilities;
	VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice->handle, swapchain.surface, &surfaceCapabilities));

	swapchain.info.imageExtent = surfaceCapabilities.currentExtent;
	swapchain.info.oldSwapchain = swapchain.handle;
	VK_CHECK(vkCreateSwapchainKHR(handle, &swapchain.info, VULKAN_ALLOCATION_CALLBACKS, &swapchain.handle));
	vkDestroySwapchainKHR(handle, swapchain.info.oldSwapchain, VULKAN_ALLOCATION_CALLBACKS);

	VK_CHECK(vkGetSwapchainImagesKHR(handle, swapchain.handle, &swapchain.numBuffers, nullptr));
	VkImage swapchainImages[MaxNumSwapChainBuffers] = { 0 };
	VK_CHECK(vkGetSwapchainImagesKHR(handle, swapchain.handle, &swapchain.numBuffers, swapchainImages));

	static VkSemaphoreCreateInfo semaphoreInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
	};
	static VkFenceCreateInfo fenceInfo = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT
	};
	for (uint32 i = 0; i < swapchain.numBuffers; i++)
	{
		VK_CHECK(vkCreateSemaphore(handle, &semaphoreInfo, VULKAN_ALLOCATION_CALLBACKS, &swapchain.imageAcquiredSemaphores[i]));
		VK_CHECK(vkCreateFence(handle, &fenceInfo, VULKAN_ALLOCATION_CALLBACKS, &swapchain.imageAcquiredFences[i]));

		VulkanTexture texture = {
			.handle = swapchainImages[i],
			.swapchainBuffer = true,
			.width = swapchain.info.imageExtent.width,
			.height = swapchain.info.imageExtent.height,
			.depth = 1,
			.arrayLayers = 1,
			.mipLevels = 1,
			.format = swapchain.info.imageFormat,
			.type = VK_IMAGE_TYPE_2D,
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.clearValue = {.color = {.float32 = { 0.0f, 0.0f, 0.0f, 0.0f } } },
		};

		uint32 textureIndex = 0;
		if (!freeTextures.empty())
		{
			textureIndex = freeTextures.back();
			freeTextures.pop_back();
			textures[textureIndex] = texture;
		}
		else
		{
			textureIndex = (uint32)textures.size();
			textures.emplace_back(texture);
		}
		swapchain.buffers[i] = backend->handleManager.Allocate<RenderBackendTextureHandle>(deviceMask);
		SetRenderBackendHandleRepresentation(swapchain.buffers[i].GetIndex(), textureIndex);
	}
	AcquireImageIndex(index);
}

RenderBackendTextureHandle VulkanDevice::GetActiveSwapChainBackBuffer(uint32 index)
{
 	return swapchains[index].buffers[swapchains[index].activeBackBufferIndex];
}

static VkSurfaceKHR CreateSurface(VulkanRenderBackend* backend, VkInstance instance, uint64 window)
{
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	VkWin32SurfaceCreateInfoKHR win32SurfaceInfo = {
		.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
		.hinstance = GetModuleHandle(NULL),
		.hwnd = (HWND)window,
	}; 
	VK_CHECK(vkCreateWin32SurfaceKHR(instance, &win32SurfaceInfo, VULKAN_ALLOCATION_CALLBACKS, &surface));
	return surface;
}

uint32 VulkanDevice::CreateSwapChain(uint64 window)
{
	VulkanSwapchain swapchain;

	VkSurfaceKHR surface = CreateSurface(backend, instance, window);
	uint32 presentQueueFamilyIndex = physicalDevice->queueFamilyIndices[(uint32)QueueFamily::Graphics];

	VkBool32 supported = VK_FALSE;
	VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice->handle, presentQueueFamilyIndex, surface, &supported));
	ASSERT(supported == VK_TRUE);

	uint32 presentModeCount = 0;
	VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice->handle, surface, &presentModeCount, nullptr));
	std::vector<VkPresentModeKHR> availablePresentModes(presentModeCount);
	VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice->handle, surface, &presentModeCount, availablePresentModes.data()));
	
	uint32 numSurfaceFormats;
	VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice->handle, surface, &numSurfaceFormats, nullptr));
	std::vector<VkSurfaceFormatKHR> availableSurfaceFormats(numSurfaceFormats);
	VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice->handle, surface, &numSurfaceFormats, availableSurfaceFormats.data()));

	VkSurfaceFormatKHR surfaceFormat;
	for (const auto& format : availableSurfaceFormats)
	{
		if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			surfaceFormat = format;
			break;
		}
	}

	VkSurfaceCapabilitiesKHR surfaceCapabilities;
	VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice->handle, surface, &surfaceCapabilities));
	ASSERT(surfaceCapabilities.supportedUsageFlags & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
	VkSurfaceTransformFlagBitsKHR preTransform;
	if (surfaceCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
	{
		preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	}
	else
	{
		preTransform = surfaceCapabilities.currentTransform;
	}
	VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
	if (surfaceCapabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
	{
		compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	}

	uint32 imageCount = std::max(std::min(2u, surfaceCapabilities.maxImageCount), surfaceCapabilities.minImageCount);

	swapchain.info = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.surface = surface,
		.minImageCount = imageCount,
		.imageFormat = surfaceFormat.format,
		.imageColorSpace = surfaceFormat.colorSpace,
		.imageExtent = surfaceCapabilities.currentExtent,
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = &presentQueueFamilyIndex,
		.preTransform = preTransform,
		.compositeAlpha = compositeAlpha,
		.presentMode = VK_PRESENT_MODE_FIFO_KHR,
		.clipped = VK_TRUE,
		.oldSwapchain = VK_NULL_HANDLE,
	};
	swapchain.surface = surface;
	VK_CHECK(vkCreateSwapchainKHR(handle, &swapchain.info, VULKAN_ALLOCATION_CALLBACKS, &swapchain.handle));

	VK_CHECK(vkGetSwapchainImagesKHR(handle, swapchain.handle, &swapchain.numBuffers, nullptr));
	VkImage swapchainImages[MaxNumSwapChainBuffers] = { 0 };
	VK_CHECK(vkGetSwapchainImagesKHR(handle, swapchain.handle, &swapchain.numBuffers, swapchainImages));

	static VkSemaphoreCreateInfo semaphoreInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
	};
	static VkFenceCreateInfo fenceInfo = { 
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT
	};
	for (uint32 i = 0; i < swapchain.numBuffers; i++)
	{
		VK_CHECK(vkCreateSemaphore(handle, &semaphoreInfo, VULKAN_ALLOCATION_CALLBACKS, &swapchain.imageAcquiredSemaphores[i]));
		VK_CHECK(vkCreateFence(handle, &fenceInfo, VULKAN_ALLOCATION_CALLBACKS, &swapchain.imageAcquiredFences[i]));

		VulkanTexture texture = {
			.handle = swapchainImages[i],
			.swapchainBuffer = true,
			.width = swapchain.info.imageExtent.width,
			.height = swapchain.info.imageExtent.height,
			.depth = 1,
			.arrayLayers = 1,
			.mipLevels = 1,
			.format = swapchain.info.imageFormat,
			.type = VK_IMAGE_TYPE_2D,
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.clearValue = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 0.0f } } },
		};

		uint32 textureIndex = 0;
		if (!freeTextures.empty())
		{
			textureIndex = freeTextures.back();
			freeTextures.pop_back();
			textures[textureIndex] = texture;
		}
		else
		{
			textureIndex = (uint32)textures.size();
			textures.emplace_back(texture);
		}
		swapchain.buffers[i] = backend->handleManager.Allocate<RenderBackendTextureHandle>(deviceMask);
		SetRenderBackendHandleRepresentation(swapchain.buffers[i].GetIndex(), textureIndex);
	}

	uint32 swapchainIndex = (uint32)swapchains.size();
	swapchains.emplace_back(swapchain);
	renderCompleteSemaphores.resize(swapchains.size());

	AcquireImageIndex(swapchainIndex);

	return swapchainIndex;
}

void VulkanDevice::DestroySwapChain(uint32 index)
{
	VulkanSwapchain& swapchain = swapchains[index];
	vkDeviceWaitIdle(handle);
	vkDestroySwapchainKHR(handle, swapchain.handle, VULKAN_ALLOCATION_CALLBACKS);
	vkDestroySurfaceKHR(instance, swapchain.surface, VULKAN_ALLOCATION_CALLBACKS);
	for (uint32 i = 0; i < swapchain.numBuffers; i++)
	{
		// vkDestroyImageView(handle, swapchain.buffers[i].view, VULKAN_ALLOCATION_CALLBACKS);
		vkDestroyFence(handle, swapchain.imageAcquiredFences[i], VULKAN_ALLOCATION_CALLBACKS);
		vkDestroySemaphore(handle, swapchain.imageAcquiredSemaphores[i], VULKAN_ALLOCATION_CALLBACKS);
	}
	swapchains.erase(swapchains.begin() + index);
}

VulkanDevice::VulkanDevice()
	: backend(nullptr)
	, physicalDevice(nullptr)
	, instance(VK_NULL_HANDLE)
	, handle(VK_NULL_HANDLE)
	, vmaAllocator(VK_NULL_HANDLE)
	, bindlessManager()
{
	ASSERT(deviceMask == 0);
	ASSERT(handle == VK_NULL_HANDLE);
}

VulkanDevice::~VulkanDevice()
{
	ASSERT(deviceMask == 0);
	ASSERT(handle == VK_NULL_HANDLE);
}

bool VulkanDevice::Init(VulkanRenderBackend* backend, VulkanPhysicalDevice* physicalDevice, const VulkanBindlessConfig& bindlessConfig)
{
	this->backend = backend;
	this->physicalDevice = physicalDevice;
	this->instance = backend->instance;
	this->deviceMask = ~uint32(0);

	// Create logical device
	{
		std::vector<const char*> requiredDeviceExtensions;
		requiredDeviceExtensions.push_back(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME);
#if DEBUG_ONLY_RAY_TRACING_ENBALE
		requiredDeviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);

		// TODO: delete these two externsions
		requiredDeviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
		requiredDeviceExtensions.push_back(VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME);
#endif
		// requiredDeviceExtensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);

		for (const auto& requiredExtension : requiredDeviceExtensions)
		{
			if (CheckInstanceExtensionSupport(requiredExtension, physicalDevice->extensionProperties))
			{
				enabledDeviceExtensions.push_back(requiredExtension);
				HE_LOG_INFO("Enabled device extension: {}.", requiredExtension);
			}
			else
			{
				HE_LOG_ERROR("The device doesn't support the required extension: {}.", requiredExtension);
				return false;
			}
		}

		std::vector<VkDeviceQueueCreateInfo> queueInfos = {};
		std::vector<float> queuePriorities[NUM_QUEUE_FAMILIES];
		for (uint32 family = 0; family < NUM_QUEUE_FAMILIES; family++)
		{
			const uint32 queueCount = numCommandQueues[family];
			// Set all priorities to 1.0 for now.
			queuePriorities[family].resize(queueCount, 1.0f);
			commandQueues[family].resize(queueCount);

			VkDeviceQueueCreateInfo queueInfo = {
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.queueFamilyIndex = physicalDevice->queueFamilyIndices[family],
				.queueCount = queueCount,
				.pQueuePriorities = queuePriorities[family].data()
			};
			if (queueInfo.queueCount > 0)
			{
				queueInfos.push_back(queueInfo);
			}
		}

		VkDeviceCreateInfo deviceInfo = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.pNext = physicalDevice->featuresEntry, // Vulkan 1.2
			.queueCreateInfoCount = (uint32)(queueInfos.size()),
			.pQueueCreateInfos = queueInfos.data(),
			.enabledExtensionCount = (uint32)(enabledDeviceExtensions.size()),
			.ppEnabledExtensionNames = enabledDeviceExtensions.data(),
			.pEnabledFeatures = &physicalDevice->enabledFeatures
		};

		VK_CHECK(vkCreateDevice(physicalDevice->handle, &deviceInfo, VULKAN_ALLOCATION_CALLBACKS, &handle));

		SetDebugUtilsObjectName(VK_OBJECT_TYPE_DEVICE, (uint64)handle, physicalDevice->properties.deviceName);
	}

	// Init command queues
	{
		for (uint32 family = 0; family < NUM_QUEUE_FAMILIES; family++)
		{
			for (uint32 queueIndex = 0; queueIndex < (uint32)commandQueues[family].size(); queueIndex++)
			{
				VkQueue queueHandle;
				vkGetDeviceQueue(handle, physicalDevice->queueFamilyIndices[family], queueIndex, &queueHandle);
				commandQueues[family][queueIndex] = {
					.handle = queueHandle,
					.familyIndex = physicalDevice->queueFamilyIndices[family],
					.queueIndex = queueIndex
				};
			}
		}
	}

	commandBufferManager = new VulkanCommandBufferManager(this, QueueFamily::Graphics);

	CreateVmaAllocator();

	CreateBindlessManager(bindlessConfig);

	CreateDefaultResources();

	return true;
}


void VulkanDevice::Shutdown()
{
	WaitIdle();
	DestroyBindlessManager();
	for (uint32 i = 0; i < (uint32)swapchains.size(); i++)
	{
		DestroySwapChain(i);
	}
	DestroyVmaAllocator();
	if (handle != VK_NULL_HANDLE)
	{
		vkDestroyDevice(handle, VULKAN_ALLOCATION_CALLBACKS);
		handle = VK_NULL_HANDLE;
	}
	deviceMask = 0;
}

bool VulkanDevice::IsDeviceExtensionEnabled(const char* extension)
{
	for (const auto& enabledExtension : enabledDeviceExtensions)
	{
		if (strcmp(extension, enabledExtension) == 0)
		{
			return true;
		}
	}
	return false;
}

void VulkanDevice::SetDebugUtilsObjectName(VkObjectType type, uint64 handle, const char* name)
{
	if (!name || !backend->IsInstanceExtensionEnabled(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
	{
		return;
	}

	VkDebugUtilsObjectNameInfoEXT nameInfo = {
		.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
		.objectType = type,
		.objectHandle = handle,
		.pObjectName = name
	};

	VK_CHECK(backend->functions.vkSetDebugUtilsObjectNameEXT(this->handle, &nameInfo));
}

bool VulkanDevice::CreateBindlessManager(const VulkanBindlessConfig& bindlessConfig)
{
	const uint32 maxNumSampledImages           = physicalDevice->descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindSampledImages;
	const uint32 maxNumSamplers                = physicalDevice->descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindSamplers;
	const uint32 maxNumStorageImage            = physicalDevice->descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindStorageImages;
	const uint32 maxNumStorageBuffers          = physicalDevice->descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindStorageBuffers;
	const uint32 maxNumAccellerationStructures = physicalDevice->accelerationStructureProperties.maxDescriptorSetAccelerationStructures;

	uint32 numSampledImages          = Math::Min(bindlessConfig.numSampledImages, maxNumSampledImages);
	uint32 numSamplers               = Math::Min(bindlessConfig.numSamplers, maxNumSamplers);
	uint32 numStorageImages          = Math::Min(bindlessConfig.numStorageImages, maxNumStorageImage);
	uint32 numStorageBuffers         = Math::Min(bindlessConfig.numStorageBuffers, maxNumStorageBuffers);
	uint32 numAccelerationStructures = Math::Min(bindlessConfig.numAccelerationStructures, maxNumAccellerationStructures);

	const VkDescriptorPoolSize bindlessPoolSizes[] = {
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,              numSampledImages          },
		{ VK_DESCRIPTOR_TYPE_SAMPLER,                    numSamplers               },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              numStorageImages          },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             numStorageBuffers         },
#if DEBUG_ONLY_RAY_TRACING_ENBALE
		{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, numAccelerationStructures }, 
#endif
	};

	uint32 poolSizeCount = ARRAY_SIZE(bindlessPoolSizes);

	VkDescriptorPoolCreateInfo descriptorPoolInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT,
		.maxSets = 1,
		.poolSizeCount = poolSizeCount,
		.pPoolSizes = bindlessPoolSizes
	};

	VkResult result = vkCreateDescriptorPool(handle, &descriptorPoolInfo, VULKAN_ALLOCATION_CALLBACKS, &bindlessManager.pool);
	if (result != VK_SUCCESS)
	{
		HE_LOG_ERROR("InitBindlessContext(): Failed to create descriptor pool.");
		return false;
	}

	const VkDescriptorSetLayoutBinding bindlessDescriptorSetLayoutBindings[] = {
		{.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,              .descriptorCount = numSampledImages,          .stageFlags = VK_SHADER_STAGE_ALL },
		{.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,                    .descriptorCount = numSamplers,               .stageFlags = VK_SHADER_STAGE_ALL },
		{.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              .descriptorCount = numStorageImages,          .stageFlags = VK_SHADER_STAGE_ALL },
		{.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             .descriptorCount = numStorageBuffers,         .stageFlags = VK_SHADER_STAGE_ALL },
#if DEBUG_ONLY_RAY_TRACING_ENBALE
		{.binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, .descriptorCount = numAccelerationStructures, .stageFlags = VK_SHADER_STAGE_ALL },
#endif		
	};
	uint32 numBindings = ARRAY_SIZE(bindlessDescriptorSetLayoutBindings);

	const VkDescriptorBindingFlagsEXT bindlessDescriptorBindingFlags[] = {
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT,
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT,
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT,
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT,
#if DEBUG_ONLY_RAY_TRACING_ENBALE
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT,
#endif		
	};
	ASSERT(numBindings == ARRAY_SIZE(bindlessDescriptorBindingFlags));

	VkDescriptorSetLayoutBindingFlagsCreateInfoEXT descriptorSetLayoutBindingFlagsInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT,
		.bindingCount = numBindings,
		.pBindingFlags = bindlessDescriptorBindingFlags
	};

	const VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = &descriptorSetLayoutBindingFlagsInfo,
		.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT,
		.bindingCount = numBindings,
		.pBindings = bindlessDescriptorSetLayoutBindings
	};
	result = vkCreateDescriptorSetLayout(handle, &descriptorSetLayoutInfo, VULKAN_ALLOCATION_CALLBACKS, &bindlessManager.layout);
	if (result != VK_SUCCESS)
	{
		HE_LOG_ERROR("InitBindlessContext(): Failed to create descriptor set layout.");
		return false;
	}

	const VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = bindlessManager.pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &bindlessManager.layout
	};
	result = vkAllocateDescriptorSets(handle, &descriptorSetAllocateInfo, &bindlessManager.set);
	if (result != VK_SUCCESS)
	{
		HE_LOG_ERROR("InitBindlessContext(): Failed to allocate descriptor set.");
		return false;
	}

	bindlessManager.config = {
		.numSampledImages = numSampledImages,
		.numSamplers = numSamplers,
		.numStorageImages = numStorageImages,
		.numStorageBuffers = numStorageBuffers,
		.numAccelerationStructures = numAccelerationStructures,
	};

	for (int32 i = numSampledImages - 1; i >= 0; i--)
	{
		bindlessManager.freeSampledImages.push_back(i);
	}
	for (int32 i = numSamplers - 1; i >= 0; i--)
	{
		bindlessManager.freeSamplers.push_back(i);
	}
	for (int32 i = numStorageImages - 1; i >= 0; i--)
	{
		bindlessManager.freeStorageImages.push_back(i);
	}
	for (int32 i = numStorageBuffers - 1; i >= 0; i--)
	{
		bindlessManager.freeStorageBuffers.push_back(i);
	}
	for (int32 i = numAccelerationStructures - 1; i >= 0; i--)
	{
		bindlessManager.freeAccelerationStructures.push_back(i);
	}

	return true;
}

void VulkanDevice::DestroyBindlessManager()
{
	if (bindlessManager.pool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPool(handle, bindlessManager.pool, VULKAN_ALLOCATION_CALLBACKS);
		bindlessManager.pool = VK_NULL_HANDLE;
	}
	if (bindlessManager.layout != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorSetLayout(handle, bindlessManager.layout, VULKAN_ALLOCATION_CALLBACKS);
		bindlessManager.layout = VK_NULL_HANDLE;
	}
}

void VulkanDevice::CreateDefaultResources()
{
	// Sampled images
	//for (uint32 i = 0; i != (uint32)TextureType::Count; i++)
	//{
	//	defaultSampledImages[i] = CreateImage();
	//	VkImageView defaultSampledImage = images[defaultSampledImage[i].GetIndex()].views[0];
	//}

	//// Sampler
	//{
	//	VkSamplerCreateInfo defaultSamplerInfo = {
	//		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
	//		.magFilter = VK_FILTER_LINEAR,
	//		.minFilter = VK_FILTER_LINEAR,
	//		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
	//		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	//		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	//		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	//		.mipLodBias = 0.f,
	//		.anisotropyEnable = VK_TRUE,
	//		.maxAnisotropy = 16.f,
	//		.compareEnable = VK_FALSE,
	//		.compareOp = VK_COMPARE_OP_NEVER,
	//		.minLod = 0.f,
	//		.maxLod = 16.f,
	//		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
	//		.unnormalizedCoordinates = VK_FALSE
	//	};
	//	defaultSampler = CreateSampler(&defaultSamplerInfo, "Default Sampler");
	//	VkSampler sampler = samplers[defaultSampler.GetIndex()].handle;
	//	VkDescriptorImageInfo sampleInfo = {
	//		.sampler = sampler,
	//		.imageView = VK_NULL_HANDLE,
	//		.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED
	//	};
	//	VkWriteDescriptorSet writeDescriptorSet = {
	//		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
	//		.dstSet = bindlessManager->set,
	//		.dstBinding = VULKAN_BINDLESS_DESCRIPTOR_SLOT_SAMPLERS,
	//		.dstArrayElement = 0,
	//		.descriptorCount = 1,
	//		.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
	//		.pImageInfo = &sampleInfo,
	//	};
	//	vkUpdateDescriptorSets(handle, 1, &writeDescriptorSet, 0, 0);
	//}

	// Storage Images
	/*for (uint32 i = 0; i != (uint32)TextureType::Count; i++)
	{
		VkImageView defaultStorageImage = images[device->resource_manager.handle_indirection[decode_index(device->null_uav_images[i])]].views[0];
		VkDescriptorImageInfo imageInfo = { 
			.sampler = VK_NULL_HANDLE, 
			.imageView = defaultStorageImage,
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL
		};
		VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = bindlessManager->set,
			.dstBinding = VULKAN_BINDLESS_DESCRIPTOR_SLOT_STORAGE_IMAGES,
			.dstArrayElement = i,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.pImageInfo = &imageInfo,
		};
		vkUpdateDescriptorSets(handle, 1, &write, 0, 0);
	}*/

	// Storage buffers
	{
		RenderBackendBufferDesc desc = RenderBackendBufferDesc::CreateByteAddress(512);
		uint32 bufferIndex = CreateBuffer(
			&desc,
			"Default Storage Buffer");
		void* data = MapBuffer(bufferIndex);
		memset(data, 0xff, 512);
		UnmapBuffer(bufferIndex);
	}
}

void VulkanDevice::CreateVmaAllocator()
{
	VmaVulkanFunctions vmaVulkanFunctions = {
		.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties,
		.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties,
		.vkAllocateMemory = vkAllocateMemory,
		.vkFreeMemory = vkFreeMemory,
		.vkMapMemory = vkMapMemory,
		.vkUnmapMemory = vkUnmapMemory,
		.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges,
		.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges,
		.vkBindBufferMemory = vkBindBufferMemory,
		.vkBindImageMemory = vkBindImageMemory,
		.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements,
		.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements,
		.vkCreateBuffer = vkCreateBuffer,
		.vkDestroyBuffer = vkDestroyBuffer,
		.vkCreateImage = vkCreateImage,
		.vkDestroyImage = vkDestroyImage,
		.vkCmdCopyBuffer = vkCmdCopyBuffer,
	};

	VmaAllocatorCreateFlags flags = 0;
	if (IsDeviceExtensionEnabled(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME) && IsDeviceExtensionEnabled(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME))
	{
		flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
		vmaVulkanFunctions.vkGetBufferMemoryRequirements2KHR = reinterpret_cast<PFN_vkGetBufferMemoryRequirements2KHR>(vkGetInstanceProcAddr(instance, "vkGetBufferMemoryRequirements2KHR"));
		vmaVulkanFunctions.vkGetImageMemoryRequirements2KHR = reinterpret_cast<PFN_vkGetImageMemoryRequirements2KHR>(vkGetInstanceProcAddr(instance, "vkGetImageMemoryRequirements2KHR"));
	}
	else
	{
		HE_LOG_WARNING("Missing device extensions: {} and {}.", VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
	}

	if (IsDeviceExtensionEnabled(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME))
	{
		flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	}

	VmaAllocatorCreateInfo allocatorInfo = {
		.flags = flags,
		.physicalDevice = physicalDevice->handle,
		.device = handle,
		.pVulkanFunctions = &vmaVulkanFunctions,
		.instance = instance,
	};

	VK_CHECK(vmaCreateAllocator(&allocatorInfo, &vmaAllocator));
}

void VulkanDevice::DestroyVmaAllocator()
{
	if (vmaAllocator != VK_NULL_HANDLE)
	{
		VmaStats stats; 
		vmaCalculateStats(vmaAllocator, &stats);
		HE_LOG_INFO("Total device memory leaked: {} bytes.", (uint32)stats.total.usedBytes);
		vmaDestroyAllocator(vmaAllocator);
		vmaAllocator = VK_NULL_HANDLE;
	}
}

void VulkanDevice::WaitIdle()
{
	VK_CHECK(vkDeviceWaitIdle(handle));
}

class VulkanRenderCommandListContext
{
public:
	VulkanRenderCommandListContext(VulkanDevice* device, QueueFamily family, VkCommandBuffer commandBuffer)
		: device(device)
		, queueFamily(family)
		, commandBuffer(commandBuffer)
		, activeRenderPass(VK_NULL_HANDLE)
		, activeComputePipeline(VK_NULL_HANDLE)
		, activeGraphicsPipeline(VK_NULL_HANDLE)
		, activeRayTracingPipeline(VK_NULL_HANDLE)
		, statistics()
		, imageBarriers()
		, bufferBarriers() {}
	virtual ~VulkanRenderCommandListContext() = default;
	inline QueueFamily GetQueueFamily() const { return queueFamily; }
	inline VkCommandBuffer GetCommandBuffer() const { return commandBuffer; }
	inline RenderStatistics GetRenderStatistics() const { return statistics; }
	bool CompileRenderCommands(const RenderCommandContainer& container);
	bool CompileRenderCommand(const RenderCommandCopyBuffer& command);
	bool CompileRenderCommand(const RenderCommandCopyTexture& command);
	bool CompileRenderCommand(const RenderCommandUpdateBuffer& command);
	bool CompileRenderCommand(const RenderCommandUpdateTexture& command);
	bool CompileRenderCommand(const RenderCommandBarriers& command);
	bool CompileRenderCommand(const RenderCommandTransitions& command);
	bool CompileRenderCommand(const RenderCommandBeginTimingQuery& command);
	bool CompileRenderCommand(const RenderCommandEndTimingQuery& command);
	bool CompileRenderCommand(const RenderCommandResetTimingQueryHeap& command);
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
	bool PrepareForDispatch(RenderBackendShaderHandle shader, const ShaderArguments& shaderArguments);
	bool PrepareForDraw(RenderBackendShaderHandle shader, PrimitiveTopology topology, RenderBackendBufferHandle indexBuffer, const ShaderArguments& shaderArguments);
	VulkanDevice* device;
	QueueFamily queueFamily;
	VkCommandBuffer commandBuffer;
	VkRenderPass activeRenderPass;
	VkPipeline activeComputePipeline;
	VkPipeline activeGraphicsPipeline;
	VkPipeline activeRayTracingPipeline;
	RenderStatistics statistics;
	std::vector<VkImageMemoryBarrier2KHR> imageBarriers;
	std::vector<VkBufferMemoryBarrier2KHR> bufferBarriers;
};

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandCopyBuffer& command)
{
	const auto& srcBuffer = device->GetBuffer(command.srcBuffer);
	const auto& dstBuffer = device->GetBuffer(command.dstBuffer);
	VkBufferCopy copyRegion = {
		.srcOffset = command.srcOffset,
		.dstOffset = command.dstOffset,
		.size = command.bytes,
	};
	vkCmdCopyBuffer(commandBuffer, srcBuffer->handle, dstBuffer->handle, 1, &copyRegion);
	return true;
}

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandCopyTexture& command)
{
	const auto& srcTexture = device->GetTexture(command.srcTexture);
	const auto& dstTexture = device->GetTexture(command.dstTexture);

	ASSERT(!HAS_ANY_FLAGS(srcTexture->flags, TextureCreateFlags::Readback));
	if (HAS_ANY_FLAGS(dstTexture->flags, TextureCreateFlags::Readback))
	{
		VkBufferImageCopy copy[MaxNumTextureMipLevels] = {};
		uint64 bufferOffset = 0;
		uint32 width = srcTexture->width;
		uint32 height = srcTexture->height;
		uint32 depth = srcTexture->depth;

		const VulkanCpuReadbackBuffer* cpuReadbackBuffer = dstTexture->cpuReadbackBuffer;
		for (uint32 mipLevel = 0; mipLevel < srcTexture->mipLevels; mipLevel++)
		{
			copy[mipLevel].bufferOffset = cpuReadbackBuffer->mipOffsets[mipLevel];
			//copy[mipLevel].bufferRowLength = width;
			//copy[mipLevel].bufferImageHeight = height;
			copy[mipLevel].imageSubresource.baseArrayLayer = command.srcSubresourceLayers.firstLayer;
			copy[mipLevel].imageSubresource.mipLevel = mipLevel;
			copy[mipLevel].imageSubresource.layerCount = 1;
			copy[mipLevel].imageSubresource.aspectMask = srcTexture->aspectMask;
			copy[mipLevel].imageExtent.width = width;
			copy[mipLevel].imageExtent.height = height;
			copy[mipLevel].imageExtent.depth = depth;

			width = std::max(1u, width / 2);
			height = std::max(1u, height / 2);
			depth = std::max(1u, depth / 2);
		}

		vkCmdCopyImageToBuffer(
			commandBuffer,
			srcTexture->handle,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			cpuReadbackBuffer->handle,
			srcTexture->mipLevels,
			copy);
	}
	else
	{
		VkImageCopy copyRegion = {
			.srcSubresource = {
				.aspectMask = srcTexture->aspectMask,
				.mipLevel = command.srcSubresourceLayers.mipLevel,
				.baseArrayLayer = command.srcSubresourceLayers.firstLayer,
				.layerCount = command.srcSubresourceLayers.arrayLayers,
			},
			.srcOffset = { command.srcOffset.x, command.srcOffset.y, command.srcOffset.z },
			.dstSubresource = {
				.aspectMask = dstTexture->aspectMask,
				.mipLevel = command.dstSubresourceLayers.mipLevel,
				.baseArrayLayer = command.dstSubresourceLayers.firstLayer,
				.layerCount = command.dstSubresourceLayers.arrayLayers,
			},
			.dstOffset = { command.dstOffset.x, command.dstOffset.y, command.dstOffset.z },
			.extent = { command.extent.width, command.extent.height, command.extent.depth },
		};
		vkCmdCopyImage(commandBuffer, srcTexture->handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstTexture->handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
	}
	return true;
}

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandUpdateBuffer& command)
{
	const auto& buffer = device->GetBuffer(command.buffer);
	vkCmdUpdateBuffer(commandBuffer, buffer->handle, command.offset, command.size, command.data);
	return true;
}

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandUpdateTexture& command)
{
	return true;
}

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandBarriers& command)
{
	return true;
}

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandTransitions& command)
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

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandBeginTimingQuery& command)
{
	const auto& timingQueryHeap = device->GetTimingQueryHeap(command.timingQueryHeap);
	uint32 queryIndex = command.region * 2 + 0;
	ASSERT(queryIndex < timingQueryHeap.maxQueryCount);
	vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timingQueryHeap.handle, queryIndex);
	return true;
}

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandEndTimingQuery& command)
{
	const auto& timingQueryHeap = device->GetTimingQueryHeap(command.timingQueryHeap);
	uint32 queryIndex = command.region * 2 + 1;
	ASSERT(queryIndex < timingQueryHeap.maxQueryCount);
	vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timingQueryHeap.handle, queryIndex);
	return true;
}

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandResetTimingQueryHeap& command)
{
	const auto& timingQueryHeap = device->GetTimingQueryHeap(command.timingQueryHeap);
	uint32 firstQuery = 2 * command.regionStart;
	uint32 queryCount = 2 * command.regionCount;
	vkCmdResetQueryPool(commandBuffer, timingQueryHeap.handle, firstQuery, queryCount);
	return true;
}

void VulkanRenderCommandListContext::ApplyTransitions()
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

bool VulkanRenderCommandListContext::PrepareForDispatch(RenderBackendShaderHandle shader, const ShaderArguments& shaderArguments)
{
	VulkanPushConstants pushConstants = {};
	for (uint32 i = 0; i < 16; i++)
	{
		if (shaderArguments.slots[i].type == 1)
		{
			VulkanTexture* texture = device->GetTexture(shaderArguments.slots[i].srvSlot.srv.texture);
			pushConstants.indices[i] = texture->srvIndex;
		}
		else if (shaderArguments.slots[i].type == 2)
		{
			VulkanTexture* texture = device->GetTexture(shaderArguments.slots[i].uavSlot.uav.texture);
			pushConstants.indices[i] = texture->uavs[shaderArguments.slots[i].uavSlot.uav.mipLevel].uavIndex;
		}
		else if (shaderArguments.slots[i].type == 3)
		{
			VulkanBuffer* buffer = device->GetBuffer(shaderArguments.slots[i].bufferSlot.handle);
			pushConstants.indices[i] = (buffer->uavIndex << 16) | (uint16)shaderArguments.slots[i].bufferSlot.offset;
		}
	}
	for (uint32 i = 0; i < 16; i++)
	{
		pushConstants.data[i] = shaderArguments.data[i];
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
	return true;
}

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandDispatch& command)
{
	if (!PrepareForDispatch(command.shader, command.shaderArguments))
	{
		return false;
	}
	vkCmdDispatch(commandBuffer, command.threadGroupCountX, command.threadGroupCountY, command.threadGroupCountZ);
	statistics.computeDispatches++;
	return true;
}

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandDispatchIndirect& command)
{
	if (!PrepareForDispatch(command.shader, command.shaderArguments))
	{
		return false;
	}
	vkCmdDispatchIndirect(commandBuffer, device->GetBuffer(command.argumentBuffer)->handle, command.argumentOffset);
	statistics.computeIndirectDispatches++;
	return true;
}

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandUpdateBottomLevelAS& command)
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

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandUpdateTopLevelAS& command)
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

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandTraceRays& command)
{
	VulkanPushConstants pushConstants = {};
	for (uint32 i = 0; i < 16; i++)
	{
		if (command.shaderArguments.slots[i].type == 1)
		{
			VulkanTexture* texture = device->GetTexture(command.shaderArguments.slots[i].srvSlot.srv.texture);
			pushConstants.indices[i] = texture->srvIndex;
		}
		else if (command.shaderArguments.slots[i].type == 2)
		{
			VulkanTexture* texture = device->GetTexture(command.shaderArguments.slots[i].uavSlot.uav.texture);
			pushConstants.indices[i] = texture->uavs[command.shaderArguments.slots[i].uavSlot.uav.mipLevel].uavIndex;
		}
		else if (command.shaderArguments.slots[i].type == 3)
		{
			VulkanBuffer* buffer = device->GetBuffer(command.shaderArguments.slots[i].bufferSlot.handle);
			pushConstants.indices[i] = (buffer->uavIndex << 16) | (uint16)command.shaderArguments.slots[i].bufferSlot.offset;
		}
		else if (command.shaderArguments.slots[i].type == 4)
		{
			VulkanRayTracingAccelerationStructure* as = device->GetAccelerationStructure(command.shaderArguments.slots[i].asSlot.handle);
			pushConstants.indices[i] = as->descriptorIndex;
		}
	}
	for (uint32 i = 0; i < 16; i++)
	{
		pushConstants.data[i] = command.shaderArguments.data[i];
	}
	const void* pushConstantsValue = &pushConstants;
	uint32 pushConstantsSize = sizeof(VulkanPushConstants);

	VulkanRayTracingPipelineState* pipelineState = device->GetRayTracingPipelineState(command.pipelineState);

	if (pipelineState->handle != activeRayTracingPipeline)
	{
		VkDescriptorSet set = device->GetBindlessGlobalSet();
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineState->pipelineLayout, 0, 1, &set, 0, nullptr);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineState->handle);
		activeComputePipeline = pipelineState->handle;
		statistics.pipelines++;
	}

	if (pushConstantsSize)
	{
		vkCmdPushConstants(commandBuffer, pipelineState->pipelineLayout, VK_SHADER_STAGE_ALL, 0, pushConstantsSize, pushConstantsValue);
	}

	VulkanBuffer* sbtBuffer = device->GetBuffer(command.shaderBindingTable);

	device->GetBackend()->functions.vkCmdTraceRaysKHR(
		commandBuffer,
		&sbtBuffer->shaderBindingTable->rayGenShaderBindingTable,
		&sbtBuffer->shaderBindingTable->missShaderBindingTable,
		&sbtBuffer->shaderBindingTable->hitShaderBindingTable,
		&sbtBuffer->shaderBindingTable->callableShaderBindingTable,
		command.width,
		command.height,
		command.depth);

	statistics.traceRayDispatches++;
	
	return true;
}

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandBeginRenderPass& command)
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

	vkCmdBeginRenderPass(commandBuffer, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);
	activeRenderPass = renderPass;
	statistics.renderPasses++;
	return true;
}

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandEndRenderPass& command)
{
	vkCmdEndRenderPass(commandBuffer);
	activeRenderPass = VK_NULL_HANDLE;
	return true;
}

bool VulkanRenderCommandListContext::PrepareForDraw(RenderBackendShaderHandle shader, PrimitiveTopology topology, RenderBackendBufferHandle indexBuffer, const ShaderArguments& shaderArguments)
{
	ASSERT(activeRenderPass != VK_NULL_HANDLE);	
	VulkanPushConstants pushConstants = {};
	for (uint32 i = 0; i < 16; i++)
	{
		if (shaderArguments.slots[i].type == 1 && shaderArguments.slots[i].srvSlot.srv.texture)
		{
			VulkanTexture* texture = device->GetTexture(shaderArguments.slots[i].srvSlot.srv.texture);
			pushConstants.indices[i] = texture->srvIndex;
		}
		else if (shaderArguments.slots[i].type == 2 && shaderArguments.slots[i].uavSlot.uav.texture)
		{
			VulkanTexture* texture = device->GetTexture(shaderArguments.slots[i].uavSlot.uav.texture);
			pushConstants.indices[i] = texture->uavs[shaderArguments.slots[i].uavSlot.uav.mipLevel].uavIndex;
		}
		else if (shaderArguments.slots[i].type == 3 && shaderArguments.slots[i].bufferSlot.handle)
		{
			VulkanBuffer* buffer = device->GetBuffer(shaderArguments.slots[i].bufferSlot.handle);
			pushConstants.indices[i] = ((buffer->uavIndex & 0xffff) << 16 ) | (shaderArguments.slots[i].bufferSlot.offset & 0xffff);
		}
	}
	for (uint32 i = 0; i < 16; i++)
	{
		pushConstants.data[i] = shaderArguments.data[i];
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

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandDraw& command)
{
	if (!PrepareForDraw(command.shader, command.topology, command.indexBuffer, command.shaderArguments))
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

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandDrawIndirect& command)
{
	if (!PrepareForDraw(command.shader, command.topology, command.indexBuffer, command.shaderArguments))
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

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandSetViewport& command)
{
	VkViewport viewports[MaxNumViewports];
	for (uint32 i = 0; i < command.numViewports; i++)
	{
		/*viewports[i] = {
			.x = command.viewports[i].x,
			.y = command.viewports[i].y,
			.width = command.viewports[i].width,
			.height = command.viewports[i].height,
			.minDepth = command.viewports[i].minDepth,
			.maxDepth = command.viewports[i].maxDepth
		};*/
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

bool VulkanRenderCommandListContext::CompileRenderCommand(const RenderCommandSetScissor& command)
{
	VkRect2D scissors[MaxNumViewports];
	for (uint32 i = 0; i < command.numScissors; i++)
	{
		scissors[i] = {
			.offset = { .x = command.scissors[i].left, .y = command.scissors[i].top },
			.extent = { .width = command.scissors[i].width, .height = command.scissors[i].height }
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
	VulkanRenderCommandListContext context(data->device, data->queueFamily, data->commandBuffer);
	if (context.CompileRenderCommands(*data->commandContainer))
	{
		data->statistics = context.GetRenderStatistics();
	}
	else
	{
		// TODO
	}
}

bool VulkanRenderCommandListContext::CompileRenderCommands(const RenderCommandContainer& container)
{
#define COMPILE_RENDER_COMMAND(command, RenderCommandStruct)                             \
	case RenderCommandStruct::Type:                                                      \
	if (!CompileRenderCommand(*reinterpret_cast<const RenderCommandStruct*>(command)))   \
	{                                                                                    \
		return false;                                                                    \
	}                                                                                    \
	break;

	for (uint32 i = 0; i < container.numCommands; i++)
	{
		switch (container.types[i])
		{
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandCopyBuffer);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandCopyTexture);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandUpdateBuffer);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandUpdateTexture);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandBarriers);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandTransitions);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandBeginTimingQuery);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandEndTimingQuery);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandResetTimingQueryHeap);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandDispatch);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandDispatchIndirect);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandUpdateBottomLevelAS);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandUpdateTopLevelAS);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandTraceRays);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandSetViewport);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandSetScissor);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandBeginRenderPass);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandEndRenderPass);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandDraw);
		COMPILE_RENDER_COMMAND(container.commands[i], RenderCommandDrawIndirect);
		}
	}
#undef COMPILE_RENDER_COMMAND
	return true;
}

static void CreateRenderDevices(void* instance, PhysicalDeviceID* physicalDeviceIDs, uint32 numPhysicalDevices, uint32* outDeviceMasks)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;

	VulkanBindlessConfig bindlessConfig = {
		.numSampledImages = 16 * 1024,
		.numSamplers = 4 * 1024,
		.numStorageImages = 16 * 1024,
		.numStorageBuffers = 8 * 1024,
		.numAccelerationStructures = 8 * 1024
	};

	for (uint32 i = 0; i < numPhysicalDevices; i++)
	{
		VulkanDevice& device = backend->devices[i];
		VulkanPhysicalDevice* physicalDevice = &backend->availablePhysicalDevices[0];
		if (device.Init(backend, physicalDevice, bindlessConfig))
		{
			outDeviceMasks[i] = device.GetDeviceMask();
		}
		else
		{
			outDeviceMasks[i] = 0;
		}
	}
}

static void DestroyRenderDevices(void* instance)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance; 
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		device.Shutdown();
	}
}

static void FlushRenderDevices(void* instance)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	VulkanDevice& device = backend->devices[0];
	vkDeviceWaitIdle(device.handle);
	/*for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		vkDeviceWaitIdle(device.handle);
	}*/
}

static void Tick(void* instance)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		device.Tick();
	}
}

static RenderBackendSwapChainHandle CreateSwapChain(void* instance, uint32 deviceMask, uint64 windowHandle)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance; 
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if (device.GetDeviceMask() & deviceMask)
		{
			uint32 index = device.CreateSwapChain(windowHandle);
			RenderBackendSwapChainHandle handle = backend->handleManager.Allocate<RenderBackendSwapChainHandle>(deviceMask);
			device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
			return handle;
		}
	}
	return RenderBackendSwapChainHandle::NullHandle;
}

static void DestroySwapChain(void* instance, RenderBackendSwapChainHandle handle)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	uint32 deviceMask = handle.GetDeviceMask();
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		} 
		uint32 index = device.GetRenderBackendHandleRepresentation(handle.GetIndex());
		device.DestroySwapChain(index);
		break;
	}
}

static void ResizeSwapChain(void* instance, RenderBackendSwapChainHandle handle, uint32* width, uint32* height)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	uint32 deviceMask = handle.GetDeviceMask();
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = device.GetRenderBackendHandleRepresentation(handle.GetIndex());
		device.ResizeSwapChain(index, width, height);
		break;
	}
}

static bool PresentSwapChain(void* instance, RenderBackendSwapChainHandle handle)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	uint32 deviceMask = handle.GetDeviceMask();
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = device.GetRenderBackendHandleRepresentation(handle.GetIndex());
		VulkanSwapchain::Status status = device.PresentSwapChain(index, &device.renderCompleteSemaphores[index], 1);
		if (status == VulkanSwapchain::Status::Success)
		{
			status = device.AcquireImageIndex(index);
		}
		if (status == VulkanSwapchain::Status::OutOfDate)
		{
			device.RecreateSwapChain(index);
		}
		else if (status == VulkanSwapchain::Status::Error)
		{
			return false;
		}
		break;
	}
	return true;
}

static RenderBackendTextureHandle GetActiveSwapChainBuffer(void* instance,  RenderBackendSwapChainHandle handle)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	uint32 deviceMask = handle.GetDeviceMask();
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = device.GetRenderBackendHandleRepresentation(handle.GetIndex());
		return device.GetActiveSwapChainBackBuffer(index);
	}
	return RenderBackendTextureHandle::NullHandle;
}

static RenderBackendBufferHandle CreateBuffer(void* instance, uint32 deviceMask, const RenderBackendBufferDesc* desc, const char* name)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	RenderBackendBufferHandle handle = backend->handleManager.Allocate<RenderBackendBufferHandle>(deviceMask);
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = device.CreateBuffer(desc, name);
		device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
	}
	return handle;
}

static void ResizeBuffer(void* instance, RenderBackendBufferHandle handle, uint64 size)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	uint32 deviceMask = handle.GetDeviceMask();
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = 0;
		if (!device.TryGetRenderBackendHandleRepresentation(handle.GetIndex(), &index))
		{
			continue;
		}
		device.ResizeBuffer(index, size);
	}
}

static void MapBuffer(void* instance, RenderBackendBufferHandle handle, void** data)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	uint32 deviceMask = handle.GetDeviceMask();
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = 0;
		if (!device.TryGetRenderBackendHandleRepresentation(handle.GetIndex(), &index))
		{
			continue;
		}
		*data = device.MapBuffer(index);
	}
}

static void UnmapBuffer(void* instance, RenderBackendBufferHandle handle)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	uint32 deviceMask = handle.GetDeviceMask();
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = 0;
		if (!device.TryGetRenderBackendHandleRepresentation(handle.GetIndex(), &index))
		{
			continue;
		}
		device.UnmapBuffer(index);
	}
}

static void DestroyBuffer(void* instance, RenderBackendBufferHandle handle)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	uint32 deviceMask = handle.GetDeviceMask();
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = 0;
		if (!device.TryGetRenderBackendHandleRepresentation(handle.GetIndex(), &index))
		{
			continue;
		}
		device.DestroyBuffer(index);
		device.RemoveRenderBackendHandleRepresentation(handle.GetIndex());
	}
}

static RenderBackendTextureHandle CreateTexture(void* instance, uint32 deviceMask, const RenderBackendTextureDesc* desc, const void* data, const char* name)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	RenderBackendTextureHandle handle = backend->handleManager.Allocate<RenderBackendTextureHandle>(deviceMask);
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = device.CreateTexture(desc, data, name);
		device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
	}
	return handle;
}

static void DestroyTexture(void* instance, RenderBackendTextureHandle handle)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	uint32 deviceMask = handle.GetDeviceMask();
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = 0;
		if (!device.TryGetRenderBackendHandleRepresentation(handle.GetIndex(), &index))
		{
			continue;
		}
		device.DestroyTexture(index);
		device.RemoveRenderBackendHandleRepresentation(handle.GetIndex());
	}
}

static RenderBackendSamplerHandle CreateSampler(void* instance, uint32 deviceMask, const RenderBackendSamplerDesc* desc, const char* name)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	RenderBackendSamplerHandle handle = backend->handleManager.Allocate<RenderBackendSamplerHandle>(deviceMask);
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = device.CreateSampler(desc, name);
		device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
	}
	return handle;
}

static void DestroySampler(void* instance, RenderBackendSamplerHandle handle)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	uint32 deviceMask = handle.GetDeviceMask();
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = 0;
		if (!device.TryGetRenderBackendHandleRepresentation(handle.GetIndex(), &index))
		{
			continue;
		}
		device.DestroySampler(index);
		device.RemoveRenderBackendHandleRepresentation(handle.GetIndex());
	}
}

static RenderBackendShaderHandle CreateShader(void* instance, uint32 deviceMask, const RenderBackendShaderDesc* desc, const char* name)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	RenderBackendShaderHandle handle = backend->handleManager.Allocate<RenderBackendShaderHandle>(deviceMask);
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = device.CreateShader(desc, name);
		device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
	}
	return handle;
}

static void DestroyShader(void* instance, RenderBackendShaderHandle handle)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	uint32 deviceMask = handle.GetDeviceMask();
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = 0;
		if (!device.TryGetRenderBackendHandleRepresentation(handle.GetIndex(), &index))
		{
			continue;
		}
		device.DestroyShader(index);
		device.RemoveRenderBackendHandleRepresentation(handle.GetIndex());
	}
}

static RenderBackendTimingQueryHeapHandle CreateTimingQueryHeap(void* instance, uint32 deviceMask, const RenderBackendTimingQueryHeapDesc* desc, const char* name)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	RenderBackendTimingQueryHeapHandle handle = backend->handleManager.Allocate<RenderBackendTimingQueryHeapHandle>(deviceMask);
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		
		VulkanTimingQueryHeap timingQueryHeap = {};
		timingQueryHeap.maxQueryCount = desc->maxRegions * 2;
		VkQueryPoolCreateInfo queryPoolInfo = {
			.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
			.queryType = VK_QUERY_TYPE_TIMESTAMP,
			.queryCount = timingQueryHeap.maxQueryCount,
		};
		VK_CHECK(vkCreateQueryPool(device.handle, &queryPoolInfo, VULKAN_ALLOCATION_CALLBACKS, &timingQueryHeap.handle));

		uint32 index = device.timingQueryHeaps.Add(timingQueryHeap);
		device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
	}
	return handle;
}

static void DestroyTimingQueryHeap(void* instance, RenderBackendTimingQueryHeapHandle timingQueryHeap)
{

}

static bool GetTimingQueryHeapResults(void* instance, RenderBackendTimingQueryHeapHandle handle, uint32 regionStart, uint32 regionCount, void* results)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	VulkanDevice& device = backend->devices[0];

	uint32 index = device.GetRenderBackendHandleRepresentation(handle.GetIndex());

	const auto& timingQueryHeap = device.timingQueryHeaps.Get(index);

	uint32 firstQuery = 2 * regionStart;
	uint32 queryCount = 2 * regionCount;

	uint64 data[2 * MaxNumTimingQueryRegions];

	VkResult result = vkGetQueryPoolResults(device.handle, timingQueryHeap.handle, firstQuery, queryCount, queryCount * sizeof(uint64), data, sizeof(uint64), VK_QUERY_RESULT_64_BIT);
	
	if (result != VK_SUCCESS)
	{
		return false;
	}

	const float timestampPeriod = device.physicalDevice->properties.limits.timestampPeriod;
	double millisecondsPerTick = 1e-6f * timestampPeriod;

	for (uint32 i = 0; i < regionCount; i++)
	{
		((float*)results)[i] = millisecondsPerTick * (double)(data[2 * i + 1] - data[2 * i]);
	}

	return true;
}

static RenderBackendOcclusionQueryHeapHandle CreateOcclusionQueryHeap(void* instance, uint32 deviceMask, const RenderBackendOcclusionQueryHeapDesc* desc, const char* name)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	RenderBackendOcclusionQueryHeapHandle handle = backend->handleManager.Allocate<RenderBackendOcclusionQueryHeapHandle>(deviceMask);
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}

		VulkanOcclusionQueryHeap occlusionQueryHeap = {};
		occlusionQueryHeap.maxQueryCount = desc->maxQueries;
		VkQueryPoolCreateInfo queryPoolInfo = {
			.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
			.queryType = VK_QUERY_TYPE_OCCLUSION,
			.queryCount = occlusionQueryHeap.maxQueryCount,
		};
		VK_CHECK(vkCreateQueryPool(device.handle, &queryPoolInfo, VULKAN_ALLOCATION_CALLBACKS, &occlusionQueryHeap.handle));

		uint32 index = device.occlusionQueryHeaps.Add(occlusionQueryHeap);
		device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
	}
	return handle;
}

static void DestroyOcclusionQueryHeap(void* instance, RenderBackendOcclusionQueryHeapHandle occlusionQueryHeap)
{

}

static RenderBackendRayTracingAccelerationStructureHandle CreateTopLevelAS(void* instance, uint32 deviceMask, const RenderBackendTopLevelASDesc* desc, const char* name)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	RenderBackendRayTracingAccelerationStructureHandle handle = backend->handleManager.Allocate<RenderBackendRayTracingAccelerationStructureHandle>(deviceMask);
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = device.CreateTopLevelAS(desc, name);
		device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
	}
	return handle;
}

static RenderBackendRayTracingAccelerationStructureHandle CreateBottomLevelAS(void* instance, uint32 deviceMask, const RenderBackendBottomLevelASDesc* desc, const char* name)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	RenderBackendRayTracingAccelerationStructureHandle handle = backend->handleManager.Allocate<RenderBackendRayTracingAccelerationStructureHandle>(deviceMask);
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		uint32 index = device.CreateBottomLevelAS(desc, name);
		device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
	}
	return handle;
}

static void SubmitRenderCommandLists(void* instance, RenderCommandList** commandLists, uint32 numCommandLists)
{
	if (!commandLists || !numCommandLists)
	{
		return;
	}

	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	RenderCommandContainer* commandContainer = commandLists[0]->GetCommandContainer();
	uint32 numCommands = commandContainer->numCommands;

	if (!numCommands)
	{
		return;
	}

	std::vector<VkSubmitInfo> submitInfos[MaxNumRenderDevices][NUM_QUEUE_FAMILIES];
	VulkanSubmitContext submitContexts[MaxNumRenderDevices] = {};

	std::vector<VkSemaphore> waitSemaphores;
	std::vector<VkPipelineStageFlags> waitDstStageMasks;
	std::vector<VkSemaphore> signalSemaphores;

	uint32 deviceMask = ~0u;
	uint32 queueFamily = (uint32)QueueFamily::Graphics;

	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		VulkanSubmitContext& submitContext = submitContexts[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		VulkanCommandBuffer* primaryCommandBuffer = device.commandBufferManager->PrepareForNextCommandBuffer();
		submitContext.completeFence = primaryCommandBuffer->fence;
		vkResetFences(device.GetHandle(), 1, &submitContext.completeFence);

		VkCommandBufferBeginInfo beginInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 
		};
		VK_CHECK(vkBeginCommandBuffer(primaryCommandBuffer->handle, &beginInfo));

		if (false)
		{
			// TODO
		}
		else 
		{
			BuildCommandBufferJobData jobData = {
				.backend = backend,
				.device = &device,
				.queueFamily = QueueFamily::Graphics,
				.commandBuffer = primaryCommandBuffer->handle,
				.commandContainer = commandLists[0]->GetCommandContainer(),
			};
			BuildCommandBuffer(&jobData);
			device.renderStatistics.Add(jobData.statistics);
		}

		vkEndCommandBuffer(primaryCommandBuffer->handle);

		if (!device.swapchains.empty())
		{
			VulkanSwapchain* swapchain = &device.swapchains[0];
			waitSemaphores.push_back(swapchain->imageAcquiredSemaphores[swapchain->semaphoreIndex]);
			waitDstStageMasks.push_back(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
			submitContext.completeSemaphore = primaryCommandBuffer->semaphore;
			device.renderCompleteSemaphores[0] = submitContext.completeSemaphore;
			signalSemaphores.push_back(submitContexts[deviceIndex].completeSemaphore);
		}

		VkSubmitInfo submitInfo = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.pNext = nullptr,
			.waitSemaphoreCount = (uint32)waitSemaphores.size(),
			.pWaitSemaphores = waitSemaphores.data(),
			.pWaitDstStageMask = waitDstStageMasks.data(),
			.commandBufferCount = 1,
			.pCommandBuffers = &(primaryCommandBuffer->handle),
			.signalSemaphoreCount = (uint32)signalSemaphores.size(),
			.pSignalSemaphores = signalSemaphores.data(),
		};
		submitInfos[deviceIndex][queueFamily].emplace_back(submitInfo);
	}

	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		/*if (submitContexts[deviceIndex] == nullptr)
		{
			continue;
		}*/
		if (deviceIndex > 0)
		{
			break;
		}

		VulkanDevice& device = backend->devices[deviceIndex];
		
		VK_CHECK(vkQueueSubmit(
			device.GetCommandQueue(queueFamily, 0)->handle,
			(uint32)submitInfos[deviceIndex][queueFamily].size(),
			submitInfos[deviceIndex][queueFamily].data(),
			submitContexts[deviceIndex].completeFence
		));
	}
}

static void GetRenderStatistics(void* instance, uint32 deviceMask, RenderStatistics* outStatistics)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}
		device.GetRenderStatistics(outStatistics);
		break;
	}
}

static RenderBackendTextureSRVHandle CreateTextureSRV(void* instance, uint32 deviceMask, const RenderBackendTextureSRVDesc* desc, const char* name)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	RenderBackendTextureSRVHandle handle = backend->handleManager.Allocate<RenderBackendTextureSRVHandle>(deviceMask);
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}

		uint32 textureIndex = 0;
		if (!device.TryGetRenderBackendHandleRepresentation(desc->texture.GetIndex(), &textureIndex))
		{
			continue;
		}

		uint32 index = device.CreateTextureSRV(textureIndex, desc, name);
		device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
	}
	return handle;
}

static int32 GetTextureSRVDescriptorIndex(void* instance, uint32 deviceMask, RenderBackendTextureHandle srv)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance; 
	VulkanDevice& device = backend->devices[0];
	uint32 textureIndex = 0;
	if (!device.TryGetRenderBackendHandleRepresentation(srv.GetIndex(), &textureIndex))
	{
		return 0;
	}
	return device.GetTextureSRVDescriptorIndex(textureIndex);
}

static RenderBackendTextureUAVHandle CreateTextureUAV(void* instance, uint32 deviceMask, const RenderBackendTextureUAVDesc* desc, const char* name)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	RenderBackendTextureUAVHandle handle = backend->handleManager.Allocate<RenderBackendTextureUAVHandle>(deviceMask);
	for (uint32 deviceIndex = 0; deviceIndex < backend->numDevices; deviceIndex++)
	{
		VulkanDevice& device = backend->devices[deviceIndex];
		if ((device.GetDeviceMask() & deviceMask) == 0)
		{
			continue;
		}

		uint32 textureIndex = 0;
		if (!device.TryGetRenderBackendHandleRepresentation(desc->texture.GetIndex(), &textureIndex))
		{
			continue;
		}

		uint32 index = device.CreateTextureUAV(textureIndex, desc, name);
		device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
	}
	return handle;
}

static int32 GetTextureUAVDescriptorIndex(void* instance, uint32 deviceMask, RenderBackendTextureHandle uav)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	VulkanDevice& device = backend->devices[0];
	uint32 textureIndex = 0;
	if (!device.TryGetRenderBackendHandleRepresentation(uav.GetIndex(), &textureIndex))
	{
		return 0;
	}
	return device.GetTextureUAVDescriptorIndex(textureIndex, 0);
}

RenderBackendRayTracingPipelineStateHandle CreateRayTracingPipelineState(void* instance, uint32 deviceMask, const RenderBackendRayTracingPipelineStateDesc* desc, const char* name)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	VulkanDevice& device = backend->devices[0];
	RenderBackendRayTracingPipelineStateHandle handle = backend->handleManager.Allocate<RenderBackendRayTracingPipelineStateHandle>(deviceMask);

	{
		uint32 numShaders = (uint32)desc->shaders.size();
		uint32 numShaderGroups = (uint32)desc->shaderGroupDescs.size();

		std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos(numShaders);
		std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroupCreateInfos(numShaderGroups);

		for (uint32 shaderIndex = 0; shaderIndex < numShaders; shaderIndex++)
		{
			shaderStageCreateInfos[shaderIndex] = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = ConvertToVkShaderStageFlagBits(desc->shaders[shaderIndex].stage),
				.pName = desc->shaders[shaderIndex].entry.c_str(),
			};
			VkShaderModuleCreateInfo shaderModuleCreateInfo = {
				.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
				.codeSize = desc->shaders[shaderIndex].code.size,
				.pCode = (uint32*)desc->shaders[shaderIndex].code.data,
			};
			VK_CHECK(vkCreateShaderModule(device.GetHandle(), &shaderModuleCreateInfo, VULKAN_ALLOCATION_CALLBACKS, &shaderStageCreateInfos[shaderIndex].module));
			device.SetDebugUtilsObjectName(VK_OBJECT_TYPE_SHADER_MODULE, (uint64)shaderStageCreateInfos[shaderIndex].module, shaderStageCreateInfos[shaderIndex].pName);
		}

		uint32 numRayGenerationShaders = 0;
		uint32 numMissShaders = 0;
		uint32 numHitGroups = 0;

		for (uint32 groupIndex = 0; groupIndex < numShaderGroups; groupIndex++)
		{
			switch (desc->shaderGroupDescs[groupIndex].type)
			{
			case RenderBackendRayTracingShaderGroupType::RayGen:
				shaderGroupCreateInfos[groupIndex] = {
					.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
					.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
					.generalShader = desc->shaderGroupDescs[groupIndex].rayGenerationShader,
					.closestHitShader = VK_SHADER_UNUSED_KHR,
					.anyHitShader = VK_SHADER_UNUSED_KHR,
					.intersectionShader = VK_SHADER_UNUSED_KHR,
				};
				numRayGenerationShaders++;
				break;
			case RenderBackendRayTracingShaderGroupType::Miss:
				shaderGroupCreateInfos[groupIndex] = {
					.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
					.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
					.generalShader = desc->shaderGroupDescs[groupIndex].missShader,
					.closestHitShader = VK_SHADER_UNUSED_KHR,
					.anyHitShader = VK_SHADER_UNUSED_KHR,
					.intersectionShader = VK_SHADER_UNUSED_KHR,
				};
				numMissShaders++;
				break;
			case RenderBackendRayTracingShaderGroupType::TrianglesHitGroup:
				shaderGroupCreateInfos[groupIndex] = {
					.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
					.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
					.closestHitShader = desc->shaderGroupDescs[groupIndex].closestHitShader,
					.anyHitShader = desc->shaderGroupDescs[groupIndex].anyHitShader,
					.intersectionShader = desc->shaderGroupDescs[groupIndex].intersectionShader,
				};
				numHitGroups++;
				break;
			default:
				INVALID_ENUM_VALUE();
				break;
			}
		}
		ASSERT(numRayGenerationShaders == 1);
		
		const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rayTracingPipelineProperties = device.GetRayTracingPipelineProperties();
		
		VkPipelineLayout pipelineLayout = device.FindOrCreatePipelineLayout(sizeof(VulkanPushConstants), RenderBackendPipelineType::RayTracing);

		VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
			.stageCount = (uint32)shaderStageCreateInfos.size(),
			.pStages = shaderStageCreateInfos.data(),
			.groupCount = (uint32)shaderGroupCreateInfos.size(),
			.pGroups = shaderGroupCreateInfos.data(),
			.maxPipelineRayRecursionDepth = Math::Min(desc->maxRayRecursionDepth, rayTracingPipelineProperties.maxRayRecursionDepth),
			.layout = pipelineLayout,
		};
		
		uint32 index = (uint32)device.rayTracingPipelineStates.size();
		VulkanRayTracingPipelineState& rayTracingPipelineState = device.rayTracingPipelineStates.emplace_back();

		rayTracingPipelineState.pipelineLayout = pipelineLayout;
		rayTracingPipelineState.numRayGenerationShaders = numRayGenerationShaders;
		rayTracingPipelineState.numMissShaders = numMissShaders;
		rayTracingPipelineState.numHitGroups = numHitGroups;

		VK_CHECK(backend->functions.vkCreateRayTracingPipelinesKHR(
			device.GetHandle(),
			VK_NULL_HANDLE,
			VK_NULL_HANDLE,
			1,
			&rayTracingPipelineCreateInfo,
			VULKAN_ALLOCATION_CALLBACKS,
			&rayTracingPipelineState.handle));

		device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
	}

	return handle;
}

RenderBackendBufferHandle CreateRayTracingShaderBindingTable(void* instance, uint32 deviceMask, const RenderBackendRayTracingShaderBindingTableDesc* desc, const char* name)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	VulkanDevice& device = backend->devices[0];	
	RenderBackendBufferHandle handle = backend->handleManager.Allocate<RenderBackendBufferHandle>(deviceMask);
	{
		VulkanRayTracingPipelineState& rayTracingPipelineState = *device.GetRayTracingPipelineState(desc->rayTracingPipelineState);

		uint32 numMissShaders = rayTracingPipelineState.numMissShaders;
		uint32 numHitGroups = rayTracingPipelineState.numHitGroups;
		uint32 numShaderGroups = 1 + numMissShaders + numHitGroups;

		const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rayTracingPipelineProperties = device.GetRayTracingPipelineProperties();

		const uint32 shaderGroupHandleSizeAligned = AlignUp(rayTracingPipelineProperties.shaderGroupHandleSize, rayTracingPipelineProperties.shaderGroupHandleAlignment);
		const uint32 shaderGroupSizeAligned = AlignUp(shaderGroupHandleSizeAligned, rayTracingPipelineProperties.shaderGroupBaseAlignment);
		
		std::vector<uint8> shaderGroupHandles(shaderGroupHandleSizeAligned * numShaderGroups);
		VK_CHECK(backend->functions.vkGetRayTracingShaderGroupHandlesKHR(device.GetHandle(), rayTracingPipelineState.handle, 0, numShaderGroups, shaderGroupHandleSizeAligned * numShaderGroups, shaderGroupHandles.data()));
		
		// TODO
		uint32 rayGenGroupStride = shaderGroupSizeAligned;
		uint32 missGroupStride = shaderGroupSizeAligned;
		uint32 hitGroupStride = shaderGroupSizeAligned;

		uint32 sbtBufferSize = rayGenGroupStride + numMissShaders * missGroupStride + numHitGroups * hitGroupStride;

		RenderBackendBufferDesc sbtBufferDesc = RenderBackendBufferDesc::CreateShaderBindingTable(sbtBufferSize);
		uint32 index = device.CreateBuffer(&sbtBufferDesc, "SBT");
		VulkanBuffer& sbtBuffer = device.buffers[index];
		uint8* sbtBufferData = reinterpret_cast<uint8*>(device.MapBuffer(index));

		for (uint32 groupIndex = 0; groupIndex < numShaderGroups; groupIndex++)
		{
			memcpy(sbtBufferData, shaderGroupHandles.data() + groupIndex * shaderGroupHandleSizeAligned, shaderGroupHandleSizeAligned);
			sbtBufferData += shaderGroupSizeAligned;
		}

		VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = sbtBuffer.handle,
		};
		VkDeviceAddress sbtBufferAddress = backend->functions.vkGetBufferDeviceAddressKHR(device.GetHandle(), &bufferDeviceAddressInfo);

		sbtBuffer.shaderBindingTable = new VulkanRayTracingShaderBindingTable();

		sbtBuffer.shaderBindingTable->rayGenShaderBindingTable = {
			.deviceAddress = sbtBufferAddress,
			.stride = rayGenGroupStride,
			.size = rayGenGroupStride
		};

		sbtBuffer.shaderBindingTable->missShaderBindingTable = {
			.deviceAddress = sbtBufferAddress + sbtBuffer.shaderBindingTable->rayGenShaderBindingTable.size,
			.stride = missGroupStride,
			.size = missGroupStride * numMissShaders
		};

		sbtBuffer.shaderBindingTable->hitShaderBindingTable = {
			.deviceAddress = sbtBufferAddress + sbtBuffer.shaderBindingTable->missShaderBindingTable.size,
			.stride = hitGroupStride,
			.size = hitGroupStride * numHitGroups
		};

		sbtBuffer.shaderBindingTable->callableShaderBindingTable = { 0, 0, 0 };

		device.SetRenderBackendHandleRepresentation(handle.GetIndex(), index);
	}
	return handle;
}

static void GetTextureReadbackData(void* instance, RenderBackendTextureHandle handle, void** data)
{
	VulkanRenderBackend* backend = (VulkanRenderBackend*)instance;
	VulkanDevice& device = backend->devices[0];

	uint32 index = device.GetRenderBackendHandleRepresentation(handle.GetIndex());
	VulkanTexture& texture = device.textures[index];

	*data = texture.cpuReadbackBuffer->data;
}

}

namespace HE
{
	RenderBackend* VulkanRenderBackendCreateBackend(int flags)
	{
		VulkanRenderBackend* vulkanBackend = new VulkanRenderBackend();
		if (!vulkanBackend->Init(flags))
		{
			delete vulkanBackend;
			return nullptr;
		}
		RenderBackend* backend = new RenderBackend();
		*backend = {
			.instance = vulkanBackend,
			.Tick = Tick,
			.CreateRenderDevices = CreateRenderDevices,
			.DestroyRenderDevices = DestroyRenderDevices,
			.FlushRenderDevices = FlushRenderDevices,
			.CreateSwapChain = CreateSwapChain,
			.DestroySwapChain = DestroySwapChain,
			.ResizeSwapChain = ResizeSwapChain,
			.PresentSwapChain = PresentSwapChain,
			.GetActiveSwapChainBuffer = GetActiveSwapChainBuffer,
			.CreateBuffer = CreateBuffer,
			.ResizeBuffer = ResizeBuffer,
			.MapBuffer = MapBuffer,
			.UnmapBuffer = UnmapBuffer,
			.DestroyBuffer = DestroyBuffer,
			.CreateTexture = CreateTexture,
			.DestroyTexture = DestroyTexture,
			.GetTextureReadbackData = GetTextureReadbackData,
			.CreateTextureSRV = CreateTextureSRV,
			.GetTextureSRVDescriptorIndex = GetTextureSRVDescriptorIndex,
			.CreateTextureUAV = CreateTextureUAV,
			.GetTextureUAVDescriptorIndex = GetTextureUAVDescriptorIndex,
			.CreateSampler = CreateSampler,
			.DestroySampler = DestroySampler,
			.CreateShader = CreateShader,
			.DestroyShader = DestroyShader,
			.CreateTimingQueryHeap = CreateTimingQueryHeap,
			.DestroyTimingQueryHeap = DestroyTimingQueryHeap,
			.GetTimingQueryHeapResults = GetTimingQueryHeapResults,
			.CreateOcclusionQueryHeap = CreateOcclusionQueryHeap,
			.DestroyOcclusionQueryHeap = DestroyOcclusionQueryHeap,
			.SubmitRenderCommandLists = SubmitRenderCommandLists,
			.GetRenderStatistics = GetRenderStatistics,
			.CreateBottomLevelAS = CreateBottomLevelAS,
			.CreateTopLevelAS = CreateTopLevelAS,
			.CreateRayTracingPipelineState = CreateRayTracingPipelineState,
			.CreateRayTracingShaderBindingTable = CreateRayTracingShaderBindingTable,
		};
		return backend;
	}

	void VulkanRenderBackendDestroyBackend(RenderBackend* backend)
	{
		RenderBackendDestroyRenderDevices(backend);
		VulkanRenderBackend* vulkanBackend = (VulkanRenderBackend*)backend->instance;
		vulkanBackend->Exit();
		delete vulkanBackend;
		delete backend;
	}
}