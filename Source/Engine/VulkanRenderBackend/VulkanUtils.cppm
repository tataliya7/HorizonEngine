module;

#include <vma/vk_mem_alloc.h>

#include "Core/CoreDefinitions.h"

export module HorizonEngine.Render.VulkanRenderBackend:Utils;

__pragma(warning(push, 0))
import <VulkanCommon.h>;
import HorizonEngine.Core;
import HorizonEngine.Render.Core;
__pragma(warning(pop))

export namespace HE
{
	inline VkBool32 ConvertToVkBool(bool b)
	{
		return b ? VK_TRUE : VK_FALSE;
	}

	inline bool IsDepthOnlyFormat(VkFormat format)
	{
		return (format == VK_FORMAT_D32_SFLOAT) || (format == VK_FORMAT_D16_UNORM);
	}

	inline bool IsDepthStencilFormat(VkFormat format)
	{
		return (IsDepthOnlyFormat(format)) || (format == VK_FORMAT_D32_SFLOAT_S8_UINT) || (format == VK_FORMAT_D24_UNORM_S8_UINT) || (format == VK_FORMAT_D16_UNORM_S8_UINT);
	}

	inline bool IsStencilFormat(VkFormat format)
	{
		return (format == VK_FORMAT_D32_SFLOAT_S8_UINT) || (format == VK_FORMAT_D24_UNORM_S8_UINT) || (format == VK_FORMAT_D16_UNORM_S8_UINT);
	}

	inline uint32 AlignUp(uint32 size, uint32 alignment)
	{
		return (size + alignment - 1) & ~(alignment - 1);
	}

	void VerifyVkResult(VkResult result, const char* vkFuntion, const char* filename, uint32 line)
	{
		if (result > 0)
		{
			HE_LOG_ERROR("Unexpected result. Code: {}.Function : {}.File : {}.Line : {}.", (int)result, vkFuntion, filename, line);
		}
		else
		{
			HE_LOG_ERROR("Vulkan function returns a runtime error. Code: {}. Function: {}. File: {}. Line: {}.", (int)result, vkFuntion, filename, line);
		}
	}

	const VkFormat gVkFormatTable[] =
	{
		VK_FORMAT_UNDEFINED,
		VK_FORMAT_R8_UNORM,
		VK_FORMAT_R8_SNORM,
		VK_FORMAT_R16_UNORM,
		VK_FORMAT_R16_SNORM,
		VK_FORMAT_R8G8_UNORM,
		VK_FORMAT_R8G8_SNORM,
		VK_FORMAT_R16G16_UNORM,
		VK_FORMAT_R16G16_SNORM,
		VK_FORMAT_R16G16B16_UNORM,
		VK_FORMAT_R16G16B16_SNORM,
		VK_FORMAT_R8G8B8A8_UNORM,
		VK_FORMAT_R8G8B8A8_SNORM,
		VK_FORMAT_R16G16B16A16_UNORM,
		VK_FORMAT_R8G8B8A8_SRGB,
		VK_FORMAT_R16_SFLOAT,
		VK_FORMAT_R16G16_SFLOAT,
		VK_FORMAT_R16G16B16_SFLOAT,
		VK_FORMAT_R16G16B16A16_SFLOAT,
		VK_FORMAT_R32_SFLOAT,
		VK_FORMAT_R32G32_SFLOAT,
		VK_FORMAT_R32G32B32_SFLOAT,
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_FORMAT_R8_SINT,
		VK_FORMAT_R8_UINT,
		VK_FORMAT_R16_SINT,
		VK_FORMAT_R16_UINT,
		VK_FORMAT_R32_SINT,
		VK_FORMAT_R32_UINT,
		VK_FORMAT_R8G8_SINT,
		VK_FORMAT_R8G8_UINT,
		VK_FORMAT_R16G16_SINT,
		VK_FORMAT_R16G16_UINT,
		VK_FORMAT_R32G32_SINT,
		VK_FORMAT_R32G32_UINT,
		VK_FORMAT_R16G16B16_SINT,
		VK_FORMAT_R16G16B16_UINT,
		VK_FORMAT_R32G32B32_SINT,
		VK_FORMAT_R32G32B32_UINT,
		VK_FORMAT_R8G8B8A8_SINT,
		VK_FORMAT_R8G8B8A8_UINT,
		VK_FORMAT_R16G16B16A16_SINT,
		VK_FORMAT_R16G16B16A16_UINT,
		VK_FORMAT_R32G32B32A32_SINT,
		VK_FORMAT_R32G32B32A32_UINT,
		VK_FORMAT_B8G8R8A8_UNORM,
		VK_FORMAT_B8G8R8A8_SRGB,
		VK_FORMAT_D32_SFLOAT,
		VK_FORMAT_D16_UNORM,
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_A2B10G10R10_UNORM_PACK32,
	};
	static_assert(ARRAY_SIZE(gVkFormatTable) == (uint32)PixelFormat::Count);

	VkCullModeFlags ConvertToVkCullModeFlags(RasterizationCullMode mode)
	{
		switch (mode)
		{
		case RasterizationCullMode::None: return VK_CULL_MODE_NONE;
		case RasterizationCullMode::Front: return VK_CULL_MODE_FRONT_BIT;
		case RasterizationCullMode::Back: return VK_CULL_MODE_BACK_BIT;
		default: INVALID_ENUM_VALUE(); return VK_CULL_MODE_FLAG_BITS_MAX_ENUM;
		}
	}

	VkStencilOpState ConvertToVkStencilOpState(const StencilOpState& stencilOpState)
	{
		VkStencilOpState state;
		state.passOp = (VkStencilOp)stencilOpState.stencilPassOp;
		state.depthFailOp = (VkStencilOp)stencilOpState.stencilDepthFailOp;
		state.failOp = (VkStencilOp)stencilOpState.stencilFailOp;
		state.compareOp = (VkCompareOp)stencilOpState.compareOp;
		state.compareMask = stencilOpState.compareMask;
		state.writeMask = stencilOpState.writeMask;
		state.reference = stencilOpState.reference;
		return state;
	}

	VkFormat ConvertToVkFormat(PixelFormat format)
	{
		return gVkFormatTable[(uint32)format];
	}

	VkImageType ConvertToVkImageType(TextureType type)
	{
		switch (type)
		{
		case TextureType::Texture1D:
			return VK_IMAGE_TYPE_1D;
		case TextureType::Texture2D:
		case TextureType::TextureCube:
			return VK_IMAGE_TYPE_2D;
		case TextureType::Texture3D:
			return VK_IMAGE_TYPE_3D;
		default:
			INVALID_ENUM_VALUE();
			return VK_IMAGE_TYPE_MAX_ENUM;
		}
	}

	VkImageViewType ConvertToVkImageViewType(TextureType type, bool isArray)
	{
		switch (type)
		{
		case TextureType::Texture1D:
			return isArray ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
		case TextureType::Texture2D:
			return isArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
		case TextureType::Texture3D:
			ASSERT(isArray == false);
			return VK_IMAGE_VIEW_TYPE_3D;
		case TextureType::TextureCube:
			return isArray ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
		default:
			INVALID_ENUM_VALUE();
			return VK_IMAGE_VIEW_TYPE_MAX_ENUM;
		}
	}

	VkFilter ConvertToVkFilter(TextureFilter filter)
	{
		switch (filter)
		{
		case TextureFilter::Nearest:
			return VK_FILTER_NEAREST;
		case TextureFilter::Linear:
			return VK_FILTER_LINEAR;
		default:
			INVALID_ENUM_VALUE();
			return VK_FILTER_MAX_ENUM;
		}
	}

	VkPrimitiveTopology ConvertToVkPrimitiveTopology(PrimitiveTopology topology)
	{
		switch (topology)
		{
		case PrimitiveTopology::PointList:
			return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
		case PrimitiveTopology::LineList:
			return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
		case PrimitiveTopology::LineStrip:
			return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
		case PrimitiveTopology::TriangleList:
			return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		case PrimitiveTopology::TriangleStrip:
			return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
		case PrimitiveTopology::TriangleFan:
			return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
		default:
			INVALID_ENUM_VALUE();
			return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
		}
	}

	VkShaderStageFlagBits ConvertToVkShaderStageFlagBits(RenderBackendShaderStage stage)
	{
		switch (stage)
		{
		case RenderBackendShaderStage::Vertex:
			return VK_SHADER_STAGE_VERTEX_BIT;
		case RenderBackendShaderStage::Pixel:
			return VK_SHADER_STAGE_FRAGMENT_BIT;
		case RenderBackendShaderStage::Compute:
			return VK_SHADER_STAGE_COMPUTE_BIT;
		case RenderBackendShaderStage::RayGen:
			return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
		case RenderBackendShaderStage::AnyHit:
			return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
		case RenderBackendShaderStage::ClosestHit:
			return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
		case RenderBackendShaderStage::Miss:
			return VK_SHADER_STAGE_MISS_BIT_KHR;
		case RenderBackendShaderStage::Intersection:
			return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
		default:
			INVALID_ENUM_VALUE();
			return VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
		}
	}

	VkAttachmentLoadOp ConvertToVkAttachmentLoadOp(RenderTargetLoadOp loadOp)
	{
		switch (loadOp)
		{
		case RenderTargetLoadOp::DontCare: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		case RenderTargetLoadOp::Load: return VK_ATTACHMENT_LOAD_OP_LOAD;
		case RenderTargetLoadOp::Clear: return VK_ATTACHMENT_LOAD_OP_CLEAR;
		default: INVALID_ENUM_VALUE(); return VK_ATTACHMENT_LOAD_OP_MAX_ENUM;
		}
	}

	VkAttachmentStoreOp ConvertToVkAttachmentStoreOp(RenderTargetStoreOp storeOp)
	{
		switch (storeOp)
		{
		case RenderTargetStoreOp::DontCare: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
		case RenderTargetStoreOp::Store: return VK_ATTACHMENT_STORE_OP_STORE;
		default: INVALID_ENUM_VALUE(); return VK_ATTACHMENT_STORE_OP_MAX_ENUM;
		}
	}

	VkImageAspectFlags GetVkImageAspectFlags(VkFormat format)
	{
		if (IsDepthOnlyFormat(format))
		{
			return VK_IMAGE_ASPECT_DEPTH_BIT;
		}
		if (IsDepthStencilFormat(format))
		{
			return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
		}
		return VK_IMAGE_ASPECT_COLOR_BIT;
	}

	VkImageUsageFlags GetVkImageUsageFlags(TextureCreateFlags flags)
	{
		VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		if (HAS_ANY_FLAGS(flags, TextureCreateFlags::UnorderedAccess))
		{
			usage |= VK_IMAGE_USAGE_STORAGE_BIT;
		}
		if (HAS_ANY_FLAGS(flags, TextureCreateFlags::ShaderResource))
		{
			usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
		}
		if (HAS_ANY_FLAGS(flags, TextureCreateFlags::DepthStencil))
		{
			usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		}
		if (HAS_ANY_FLAGS(flags, TextureCreateFlags::InputAttachment))
		{
			usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
		}
		if (HAS_ANY_FLAGS(flags, TextureCreateFlags::RenderTarget))
		{
			usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		}
		return usage;
	}

	VkBufferUsageFlags GetVkBufferUsageFlags(BufferCreateFlags flags)
	{
		VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::CopySrc))
		{
			usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		}
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::CopyDst))
		{
			usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		}
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::VertexBuffer))
		{
			usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		}
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::IndexBuffer))
		{
			usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
		}
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::UniformBuffer))
		{
			usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
		}
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::IndirectArguments))
		{
			usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
		}
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::UnorderedAccess))
		{
			usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
		}
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::AccelerationStruture))
		{
			usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
		}
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::ShaderBindingTable))
		{
			usage |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
		}
		return usage;
	}

	VmaMemoryUsage GetVmaMemoryUsage(BufferCreateFlags flags)
	{
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::CpuOnly))
		{
			return VMA_MEMORY_USAGE_CPU_ONLY;
		}
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::GpuOnly))
		{
			return VMA_MEMORY_USAGE_GPU_ONLY;
		}
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::CpuToGpu))
		{
			return VMA_MEMORY_USAGE_CPU_TO_GPU;
		}
		if (HAS_ANY_FLAGS(flags, BufferCreateFlags::GpuToCpu))
		{
			return VMA_MEMORY_USAGE_GPU_TO_CPU;
		}
		return VMA_MEMORY_USAGE_GPU_TO_CPU;
	}

	VkSamplerAddressMode ConvertToVkSamplerAddressMode(TextureAddressMode addressMode)
	{
		switch (addressMode)
		{
		case TextureAddressMode::Warp: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
		case TextureAddressMode::Mirror: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
		case TextureAddressMode::Clamp: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		case TextureAddressMode::Border: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		default: INVALID_ENUM_VALUE(); return VK_SAMPLER_ADDRESS_MODE_REPEAT;
		}
	}

	VkBuildAccelerationStructureFlagsKHR ConvertToVkBuildAccelerationStructureFlagsKHR(RenderBackendAccelerationStructureBuildFlags flags)
	{
		VkBuildAccelerationStructureFlagsKHR result = 0;
		switch (flags)
		{
		case RenderBackendAccelerationStructureBuildFlags::AllowUpdate:
			result |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
			break;
		case RenderBackendAccelerationStructureBuildFlags::AllowCompaction:
			result |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
			break;
		case RenderBackendAccelerationStructureBuildFlags::PreferFastTrace:
			result |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
			break;
		case RenderBackendAccelerationStructureBuildFlags::PreferFastBuild:
			result |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
			break;
		case RenderBackendAccelerationStructureBuildFlags::MinimizeMemory:
			result |= VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;
			break;
		default:
			INVALID_ENUM_VALUE();
			break;
		}
		return result;
	}

	VkGeometryInstanceFlagsKHR ConvertToVkGeometryInstanceFlagsKHR(RenderBackendRayTracingInstanceFlags flags)
	{
		VkGeometryInstanceFlagsKHR result = 0;
		switch (flags)
		{
		case RenderBackendRayTracingInstanceFlags::TriangleFacingCullDisable:
			result |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
			break;
		case RenderBackendRayTracingInstanceFlags::TriangleFrontCounterclockwise:
			result |= VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR;
			break;
		case RenderBackendRayTracingInstanceFlags::ForceOpaque:
			result |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
			break;
		case RenderBackendRayTracingInstanceFlags::ForceNoOpaque:
			result |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
			break;
		default:
			INVALID_ENUM_VALUE();
			break;
		}
		return result;
	}

	VkGeometryFlagsKHR ConvertToVkGeometryFlagsKHR(RenderBackendGeometryFlags flags)
	{
		VkGeometryFlagsKHR result = 0;
		switch (flags)
		{
		case RenderBackendGeometryFlags::Opaque:
			result |= VK_GEOMETRY_OPAQUE_BIT_KHR;
			break;
		case RenderBackendGeometryFlags::NoDuplicateAnyHitInvocation:
			result |= VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
			break;
		default:
			INVALID_ENUM_VALUE();
			break;
		}
		return result;
	}
	
	VkCompareOp ConvertToVkCompareOp(CompareOp compareOp)
	{
		return (VkCompareOp)compareOp;
	}

	void GetVkFilterAndVkSamplerMipmapMode(
		Filter filter,
		VkFilter* outMinFilter,
		VkFilter* outMagFilter,
		VkSamplerMipmapMode* outMipmapMode,
		bool* outAnisotropyEnable,
		bool* outCompareEnable)
	{
		switch (filter)
		{
		case Filter::MinMagMipPoint:
		case Filter::MinimumMinMagMipPoint:
		case Filter::MaximumMinMagMipPoint:
			*outMinFilter = VK_FILTER_NEAREST;
			*outMagFilter = VK_FILTER_NEAREST;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			*outAnisotropyEnable = false;
			*outCompareEnable = false;
			break;
		case Filter::MinMagPointMipLinear:
		case Filter::MinimumMinMagPointMipLinear:
		case Filter::MaximumMinMagPointMipLinear:
			*outMinFilter = VK_FILTER_NEAREST;
			*outMagFilter = VK_FILTER_NEAREST;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			*outAnisotropyEnable = false;
			*outCompareEnable = false;
			break;
		case Filter::MinPointMagLinearMipPoint:
		case Filter::MinimumMinPointMagLinearMipPoint:
		case Filter::MaximumMinPointMagLinearMipPoint:
			*outMinFilter = VK_FILTER_NEAREST;
			*outMagFilter = VK_FILTER_LINEAR;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			*outAnisotropyEnable = false;
			*outCompareEnable = false;
			break;
		case Filter::MinPointMagMipLinear:
		case Filter::MinimumMinPointMagMipLinear:
		case Filter::MaximumMinPointMagMipLinear:
			*outMinFilter = VK_FILTER_NEAREST;
			*outMagFilter = VK_FILTER_LINEAR;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			*outAnisotropyEnable = false;
			*outCompareEnable = false;
			break;
		case Filter::MinLinearMagMipPoint:
		case Filter::MinimumMinLinearMagMipPoint:
		case Filter::MaximumMinLinearMagMipPoint:
			*outMinFilter = VK_FILTER_LINEAR;
			*outMagFilter = VK_FILTER_NEAREST;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			*outAnisotropyEnable = false;
			*outCompareEnable = false;
			break;
		case Filter::MinLinearMagPointMipLinear:
		case Filter::MinimumMinLinearMagPointMipLinear:
		case Filter::MaximumMinLinearMagPointMipLinear:
			*outMinFilter = VK_FILTER_LINEAR;
			*outMagFilter = VK_FILTER_NEAREST;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			*outAnisotropyEnable = false;
			*outCompareEnable = false;
			break;
		case Filter::MinMagLinearMipPoint:
		case Filter::MinimumMinMagLinearMipPoint:
		case Filter::MaximumMinMagLinearMipPoint:
			*outMinFilter = VK_FILTER_LINEAR;
			*outMagFilter = VK_FILTER_LINEAR;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			*outAnisotropyEnable = false;
			*outCompareEnable = false;
			break;
		case Filter::MinMagMipLinear:
		case Filter::MinimumMinMagMipLinear:
		case Filter::MaximumMinMagMipLinear:
			*outMinFilter = VK_FILTER_LINEAR;
			*outMagFilter = VK_FILTER_LINEAR;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			*outAnisotropyEnable = false;
			*outCompareEnable = false;
			break;
		case Filter::Anisotropic:
		case Filter::MinimumAnisotropic:
		case Filter::MaximumAnisotropic:
			*outMinFilter = VK_FILTER_LINEAR;
			*outMagFilter = VK_FILTER_LINEAR;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			*outAnisotropyEnable = true;
			*outCompareEnable = false;
			break;
		case Filter::ComparisonMinMagMipPoint:
			*outMinFilter = VK_FILTER_NEAREST;
			*outMagFilter = VK_FILTER_NEAREST;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			*outAnisotropyEnable = false;
			*outCompareEnable = true;
			break;
		case Filter::ComparisonMinMagPointMipLinear:
			*outMinFilter = VK_FILTER_NEAREST;
			*outMagFilter = VK_FILTER_NEAREST;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			*outAnisotropyEnable = false;
			*outCompareEnable = true;
			break;
		case Filter::ComparisonMinPointMagLinearMipPoint:
			*outMinFilter = VK_FILTER_NEAREST;
			*outMagFilter = VK_FILTER_LINEAR;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			*outAnisotropyEnable = false;
			*outCompareEnable = true;
			break;
		case Filter::ComparisonMinPointMagMipLinear:
			*outMinFilter = VK_FILTER_NEAREST;
			*outMagFilter = VK_FILTER_NEAREST;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			*outAnisotropyEnable = false;
			*outCompareEnable = true;
			break;
		case Filter::ComparisonMinLinearMagMipPoint:
			*outMinFilter = VK_FILTER_LINEAR;
			*outMagFilter = VK_FILTER_NEAREST;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			*outAnisotropyEnable = false;
			*outCompareEnable = true;
			break;
		case Filter::ComparisonMinLinearMagPointMipLinear:
			*outMinFilter = VK_FILTER_LINEAR;
			*outMagFilter = VK_FILTER_NEAREST;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			*outAnisotropyEnable = false;
			*outCompareEnable = true;
			break;
		case Filter::ComparisonMinMagLinearMipPoint:
			*outMinFilter = VK_FILTER_LINEAR;
			*outMagFilter = VK_FILTER_LINEAR;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			*outAnisotropyEnable = false;
			*outCompareEnable = true;
			break;
		case Filter::ComparisonMinMagMipLinear:
			*outMinFilter = VK_FILTER_LINEAR;
			*outMagFilter = VK_FILTER_LINEAR;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			*outAnisotropyEnable = false;
			*outCompareEnable = true;
			break;
		case Filter::ComparisonAnisotropic:
			*outMinFilter = VK_FILTER_LINEAR;
			*outMagFilter = VK_FILTER_LINEAR;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			*outAnisotropyEnable = true;
			*outCompareEnable = true;
			break;
		default:
			*outMinFilter = VK_FILTER_NEAREST;
			*outMagFilter = VK_FILTER_NEAREST;
			*outMipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			*outAnisotropyEnable = false;
			*outCompareEnable = false;
			break;
		}
	}

	void GetBarrierInfo2(
		RenderBackendResourceState srcState,
		RenderBackendResourceState dstState,
		VkImageLayout* outOldLayout,
		VkImageLayout* outNewLayout,
		VkPipelineStageFlags2* outSrcStageMask,
		VkPipelineStageFlags2* outDstStageMask,
		VkAccessFlags2* outSrcAccessMask,
		VkAccessFlags2* outDstAccessMask)
	{
		switch (srcState)
		{
		case RenderBackendResourceState::Undefined:
			if (outOldLayout)
			{
				*outOldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			}
			*outSrcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			*outSrcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
			break;
		case RenderBackendResourceState::VertexBuffer:
		case RenderBackendResourceState::IndexBuffer:
		case RenderBackendResourceState::ShaderResource:
			if (outOldLayout)
			{
				*outOldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			}
			*outSrcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			*outSrcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
			break;
		case RenderBackendResourceState::Present:
			if (outOldLayout)
			{
				*outOldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			}
			*outSrcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
			*outSrcAccessMask = VK_ACCESS_2_NONE;
			break;
		case RenderBackendResourceState::RenderTarget:
			if (outOldLayout)
			{
				*outOldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			}
			*outSrcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
			*outSrcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
			break;
		case RenderBackendResourceState::DepthStencil:
			if (outOldLayout)
			{
				*outOldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			}
			*outSrcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
			*outSrcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			break;
		case RenderBackendResourceState::CopySrc:
			if (outOldLayout)
			{
				*outOldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			}
			*outSrcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
			*outSrcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
			break;
		case RenderBackendResourceState::CopyDst:
			if (outOldLayout)
			{
				*outOldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			}
			*outSrcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
			*outSrcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
			break;
		case RenderBackendResourceState::UnorderedAccess:
			if (outOldLayout)
			{
				*outOldLayout = VK_IMAGE_LAYOUT_GENERAL;
			}
			*outSrcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			*outSrcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;;
			break;
		case RenderBackendResourceState::IndirectArgument:
			if (outOldLayout)
			{
				*outOldLayout = VK_IMAGE_LAYOUT_GENERAL;
			}
			*outSrcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
			*outSrcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
			break;
		default: INVALID_ENUM_VALUE(); return;
		}

		switch (dstState)
		{
		case RenderBackendResourceState::Undefined:
			if (outNewLayout)
			{
				*outNewLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			}
			*outDstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			*outDstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
			break;
		case RenderBackendResourceState::VertexBuffer:
		case RenderBackendResourceState::IndexBuffer:
		case RenderBackendResourceState::ShaderResource:
			if (outNewLayout)
			{
				*outNewLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			}
			*outDstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			*outDstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
			break;
		case RenderBackendResourceState::Present:
			if (outNewLayout)
			{
				*outNewLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			}
			*outDstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
			*outDstAccessMask = VK_ACCESS_2_NONE;
			break;
		case RenderBackendResourceState::RenderTarget:
			if (outNewLayout)
			{
				*outNewLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			}
			*outDstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
			*outDstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
			break;
		case RenderBackendResourceState::DepthStencil:
			if (outNewLayout)
			{
				*outNewLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			}
			*outDstStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
			*outDstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			break;
		case RenderBackendResourceState::DepthStencilReadOnly:
			if (outNewLayout)
			{
				*outNewLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
			}
			*outDstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			*outDstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
			break;
		case RenderBackendResourceState::CopySrc:
			if (outNewLayout)
			{
				*outNewLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			}
			*outDstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
			*outDstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
			break;
		case RenderBackendResourceState::CopyDst:
			if (outNewLayout)
			{
				*outNewLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			}
			*outDstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
			*outDstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
			break;
		case RenderBackendResourceState::UnorderedAccess:
			if (outNewLayout)
			{
				*outNewLayout = VK_IMAGE_LAYOUT_GENERAL;
			}
			*outDstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			*outDstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
			break;
		case RenderBackendResourceState::IndirectArgument:
			if (outNewLayout)
			{
				*outNewLayout = VK_IMAGE_LAYOUT_GENERAL;
			}
			*outDstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
			*outDstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
			break;
		default: INVALID_ENUM_VALUE();
			return;
		}
	}
}
