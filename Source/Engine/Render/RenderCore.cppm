module;

#include <vector>

export module HorizonEngine.Render.Core;

export import "RenderCoreDefinitions.h";

#define HE_MGPU 1

import HorizonEngine.Core;

export namespace HE
{

	/** Pixel formats. */
	enum class PixelFormat : uint32
	{
		Unknown,
		// Norm
		R8Unorm,
		R8Snorm,
		R16Unorm,
		R16Snorm,
		RG8Unorm,
		RG8Snorm,
		RG16Unorm,
		RG16Snorm,
		RGB16Unorm,
		RGB16Snorm,
		RGBA8Unorm,
		RGBA8Snorm,
		RGBA16Unorm,
		// UnormSrgb
		RGBA8UnormSrgb,
		// Float
		R16Float,
		RG16Float,
		RGB16Float,
		RGBA16Float,
		R32Float,
		RG32Float,
		RGB32Float,
		RGBA32Float,
		// Int
		R8Int,
		R8Uint,
		R16Int,
		R16Uint,
		R32Int,
		R32Uint,
		RG8Int,
		RG8Uint,
		RG16Int,
		RG16Uint,
		RG32Int,
		RG32Uint,
		RGB16Int,
		RGB16Uint,
		RGB32Int,
		RGB32Uint,
		RGBA8Int,
		RGBA8Uint,
		RGBA16Int,
		RGBA16Uint,
		RGBA32Int,
		RGBA32Uint,
		// BGRA
		BGRA8Unorm,
		BGRA8UnormSrgb,
		// Depth stencil
		D32Float,
		D16Unorm,
		D24UnormS8Uint,

		A2BGR10Unorm,
		// Count
		Count
	};

	/** Pixel format type. */
	enum class PixelFormatType
	{
		Unknown,        ///< Unknown
		Float,          ///< Floating-point
		Sint,           ///< Signed integer
		Uint,           ///< Unsigned integer
		Snorm,          ///< Signed normalized
		Unorm,          ///< Unsigned normalized
		UnormSrgb,      ///< Unsigned normalized sRGB
	};

	/** Pixel format description. */
	struct PixelFormatDesc
	{
		PixelFormat format;
		const char* name;
		PixelFormatType type;
		uint32 bytes;
		uint32 channels;
		struct
		{
			bool isDepth;
			bool isStencil;
		};
		uint32 channelBits[4];
	};

	extern const PixelFormatDesc GPixelFormatTable[];

	FORCEINLINE bool IsDepthOnlyPixelFormat(PixelFormat format)
	{
		return GPixelFormatTable[(uint32)format].isDepth && !GPixelFormatTable[(uint32)format].isStencil;
	}

	FORCEINLINE bool IsDepthStencilPixelFormat(PixelFormat format)
	{
		return GPixelFormatTable[(uint32)format].isDepth || GPixelFormatTable[(uint32)format].isStencil;
	}

	FORCEINLINE PixelFormatType GetPixelFormatType(PixelFormat format)
	{
		return GPixelFormatTable[(uint32)format].type;
	}

	FORCEINLINE uint32 GetPixelFormatBytes(PixelFormat format)
	{
		return GPixelFormatTable[(uint32)format].bytes;
	}

	FORCEINLINE const PixelFormatDesc& GetPixelFormatDesc(PixelFormat format)
	{
		return GPixelFormatTable[(uint32)format];
	}

	/**
	 * @see: https://pcisig.com/membership/member-companies
	 */
	enum class GpuVendorID
	{
		Unknown = 0,
		NIVIDIA = 0x10DE,
		AMD = 0x1002,
	};

	enum
	{
		MaxNumGPUs = 16,
		MaxNumRenderDevices = 64,
		MaxNumSwapChainBuffers = 4,
		MaxNumSimultaneousColorRenderTargets = 8,
		MaxNumViewports = 8,
		MaxNumShaderStages = 8,
		MaxNumTextureMipLevels = 14,
		MaxNumTimingQueryRegions = 128,
	};

	class RenderBackendHandle
	{
	public:
		RenderBackendHandle() = default;
		RenderBackendHandle(uint64 value) : index((uint32)(value >> 32)), deviceMask((uint32)value) {}
		RenderBackendHandle(uint32 index, uint32 deviceMask) : index(index), deviceMask(deviceMask) {}
		FORCEINLINE bool IsNullHandle()  const { return index == InvalidIndex; }
		FORCEINLINE operator bool() const { return !IsNullHandle(); }
		FORCEINLINE bool operator==(const RenderBackendHandle& rhs) const { return ((index == rhs.index) && (deviceMask == rhs.deviceMask)); }
		FORCEINLINE bool operator!=(const RenderBackendHandle& rhs) const { return ((index != rhs.index) || (deviceMask != rhs.deviceMask)); }
		FORCEINLINE RenderBackendHandle& operator=(const RenderBackendHandle& rhs) = default;
		FORCEINLINE RenderBackendHandle& operator++() { index++; return *this; }
		FORCEINLINE RenderBackendHandle& operator--() { index--; return *this; }
		FORCEINLINE uint32 GetIndex() const { return index; }
		FORCEINLINE uint32 GetDeviceMask() const { return deviceMask; }
	private:
		static const uint32 InvalidIndex = std::numeric_limits<uint32>::max();
		uint32 index = InvalidIndex;
		uint32 deviceMask = 0;
	};

	template<typename ObjectType>
	class RenderBackendHandleTyped : public RenderBackendHandle
	{
	public:
		static const RenderBackendHandleTyped NullHandle;
		RenderBackendHandleTyped() = default;
		RenderBackendHandleTyped(uint64 value) : RenderBackendHandle(value) {}
		RenderBackendHandleTyped(uint32 index, uint32 deviceMask) : RenderBackendHandle(index, deviceMask) {}
	};

	template<typename ObjectType>
	const RenderBackendHandleTyped<ObjectType> RenderBackendHandleTyped<ObjectType>::NullHandle = RenderBackendHandleTyped<ObjectType>();

	class RenderBackendSwapChain;
	using RenderBackendSwapChainHandle = RenderBackendHandleTyped<RenderBackendSwapChain>;

	class RenderBackendTexture;
	using RenderBackendTextureHandle = RenderBackendHandleTyped<RenderBackendTexture>;

	class RenderBackendTextureSRV;
	using RenderBackendTextureSRVHandle = RenderBackendHandleTyped<RenderBackendTextureSRV>;

	class RenderBackendTextureUAV;
	using RenderBackendTextureUAVHandle = RenderBackendHandleTyped<RenderBackendTextureUAV>;

	class RenderBackendBuffer;
	using RenderBackendBufferHandle = RenderBackendHandleTyped<RenderBackendBuffer>;

	class RenderBackendSampler;
	using RenderBackendSamplerHandle = RenderBackendHandleTyped<RenderBackendSampler>;

	class RenderBackendShader;
	using RenderBackendShaderHandle = RenderBackendHandleTyped<RenderBackendShader>;

	class RenderBackendTimingQueryHeap;
	using RenderBackendTimingQueryHeapHandle = RenderBackendHandleTyped<RenderBackendTimingQueryHeap>;

	class RenderBackendOcclusionQueryHeap;
	using RenderBackendOcclusionQueryHeapHandle = RenderBackendHandleTyped<RenderBackendOcclusionQueryHeap>;

	class RenderBackendAccelerationStructure;
	using RenderBackendRayTracingAccelerationStructureHandle = RenderBackendHandleTyped<RenderBackendAccelerationStructure>;

	class RenderBackendRayTracingPipelineState;
	using RenderBackendRayTracingPipelineStateHandle = RenderBackendHandleTyped<RenderBackendRayTracingPipelineState>;

	enum class QueueFamily
	{
		/// Copy queue
		Copy = 0,
		/// Asynchronous compute queue
		Compute = 1,
		/// Graphics queue
		Graphics = 2,
		Count
	};
	static_assert((uint32)QueueFamily::Graphics == 2, "Default graphics queue index should be 2.");
	static_assert((uint32)QueueFamily::Count == NUM_QUEUE_FAMILIES);

	enum class RenderBackendPipelineType
	{
		Graphics,
		Compute,
		RayTracing,
	};

	enum class RenderBackendShaderStage
	{
		Vertex = 0,
		Pixel = 1,
		Compute = 2,
		RayGen = 3,
		AnyHit = 4,
		ClosestHit = 5,
		Miss = 6,
		Intersection = 7,
		Count,
	};
	ENUM_CLASS_OPERATORS(RenderBackendShaderStage);

	enum class ShaderStageFlags
	{
		None = 0,
		Vertex = (1 << (int)RenderBackendShaderStage::Vertex),
		Pixel = (1 << (int)RenderBackendShaderStage::Pixel),
		Compute = (1 << (int)RenderBackendShaderStage::Compute),
		RayGen = (1 << (int)RenderBackendShaderStage::RayGen),
		AnyHit = (1 << (int)RenderBackendShaderStage::AnyHit),
		ClosestHit = (1 << (int)RenderBackendShaderStage::ClosestHit),
		Miss = (1 << (int)RenderBackendShaderStage::Miss),
		Intersection = (1 << (int)RenderBackendShaderStage::Intersection),
		All = ~0,
	};
	ENUM_CLASS_OPERATORS(ShaderStageFlags);

	struct ShaderBlob
	{
		uint64 size;
		uint8* data;
	};

	enum class PrimitiveTopology
	{
		PointList = 0,
		LineList = 1,
		LineStrip = 2,
		TriangleList = 3,
		TriangleStrip = 4,
		TriangleFan = 5,
	};

	enum class RasterizationCullMode
	{
		None,
		Front,
		Back,
	};

	enum class RasterizationFillMode
	{
		Wireframe,
		Solid,
	};

	enum class StencilOp
	{
		Keep = 0,
		Zero = 1,
		Replace = 2,
		IncreaseAndClamp = 3,
		DecreaseAndClamp = 4,
		Invert = 5,
		IncreaseAndWrap = 6,
		DecreaseAndWrap = 7,
	};

	enum class CompareOp
	{
		Never = 0,
		Less = 1,
		Equal = 2,
		LessOrEqual = 3,
		Greater = 4,
		NotEqual = 5,
		GreaterOrEqual = 6,
		Always = 7,
	};

	enum class BlendOp
	{
		Add = 0,
		Subtract = 1,
		ReverseSubtract = 2,
		Min = 3,
		Max = 4,
	};

	enum class BlendFactor
	{
		Zero = 0,
		One = 1,
		SrcColor = 2,
		OneMinusSrcColor = 3,
		DstColor = 4,
		OneMinusDstColor = 5,
		SrcAlpha = 6,
		OneMinusSrcAlpha = 7,
		DstAlpha = 8,
		OneMinusDstAlpha = 9,
		ConstantColor = 10,
		OneMinusConstantColor = 11,
		ConstantAlpha = 12,
		OneMinusConstantAlpha = 13,
		SrcAlphaSaturate = 14,
		Src1Color = 15,
		OneMinusSrc1Color = 16,
		Src1Alpha = 17,
		OneMinusSrc1Alpha = 18,
	};

	enum class ColorComponentFlags
	{
		R = (1 << 0),
		G = (1 << 1),
		B = (1 << 2),
		A = (1 << 3),
		All = R | G | B | A,
	};

	struct RasterizationState
	{
		RasterizationCullMode cullMode = RasterizationCullMode::None;
		RasterizationFillMode fillMode = RasterizationFillMode::Solid;
		bool  frontFaceCounterClockwise = false;
		bool  depthClampEnable = false;
		float depthBiasConstantFactor = 1.0f;
		float depthBiasSlopeFactor = 1.0f;
	};

	struct StencilOpState
	{
		StencilOp stencilPassOp = {};
		StencilOp stencilDepthFailOp = {};
		StencilOp stencilFailOp = {};
		CompareOp compareOp = {};
		uint32    compareMask = 0;
		uint32    writeMask = 0;
		uint32    reference = 0;
	};

	struct DepthStencilState
	{
		bool           depthTestEnable = false;
		bool           depthWriteEnable = false;
		CompareOp      depthCompareOp = CompareOp::Never;
		bool           stencilTestEnable = false;
		StencilOpState front = {};
		StencilOpState back = {};
	};

	struct ColorBlendAttachmentState
	{
		bool                 blendEnable = false;
		BlendFactor          srcColorBlendFactor = BlendFactor::Zero;
		BlendFactor          dstColorBlendFactor = BlendFactor::Zero;
		BlendOp              colorBlendOp = BlendOp::Add;
		BlendFactor          srcAlphaBlendFactor = BlendFactor::Zero;
		BlendFactor          dstAlphaBlendFactor = BlendFactor::Zero;
		BlendOp              alphaBlendOp = BlendOp::Add;
		ColorComponentFlags  colorWriteMask = ColorComponentFlags::All;
	};

	struct ColorBlendState
	{
		float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
		uint32 numColorAttachments = 0;
		ColorBlendAttachmentState attachmentStates[MaxNumSimultaneousColorRenderTargets];
	};

	enum class RenderTargetLoadOp
	{
		DontCare,
		Load,
		Clear,
		Count
	};

	enum class RenderTargetStoreOp
	{
		DontCare,
		Store,
		Count
	};

	enum class ClearDepthStencil
	{
		Depth,
		Stencil,
		DepthStencil,
	};

	enum class TextureType
	{
		Texture1D,
		Texture2D,
		Texture3D,
		TextureCube,
		Count
	};

	enum class TextureFilter
	{
		Nearest,
		Linear,
	};

	enum class QueryType
	{
		Occlusion,
		Timestamp,
	};

	enum class TextureCreateFlags
	{
		None = 0,
		RenderTarget = (1 << 0),
		InputAttachment = (1 << 1),
		DepthStencil = (1 << 2),
		ShaderResource = (1 << 3),
		UnorderedAccess = (1 << 4),
		Present = (1 << 5),
		SRGB = (1 << 6),
		NoTilling = (1 << 7),
		Dynamic = (1 << 8),
		Readback = (1 << 9),
	};
	ENUM_CLASS_OPERATORS(TextureCreateFlags);

	enum class TextureAddressMode
	{
		Warp = 0,
		Mirror = 1,
		Clamp = 2,
		Border = 3,
	};

	enum class Filter
	{
		MinMagMipPoint,
		MinMagPointMipLinear,
		MinPointMagLinearMipPoint,
		MinPointMagMipLinear,
		MinLinearMagMipPoint,
		MinLinearMagPointMipLinear,
		MinMagLinearMipPoint,
		MinMagMipLinear,
		Anisotropic,
		ComparisonMinMagMipPoint,
		ComparisonMinMagPointMipLinear,
		ComparisonMinPointMagLinearMipPoint,
		ComparisonMinPointMagMipLinear,
		ComparisonMinLinearMagMipPoint,
		ComparisonMinLinearMagPointMipLinear,
		ComparisonMinMagLinearMipPoint,
		ComparisonMinMagMipLinear,
		ComparisonAnisotropic,
		MinimumMinMagMipPoint,
		MinimumMinMagPointMipLinear,
		MinimumMinPointMagLinearMipPoint,
		MinimumMinPointMagMipLinear,
		MinimumMinLinearMagMipPoint,
		MinimumMinLinearMagPointMipLinear,
		MinimumMinMagLinearMipPoint,
		MinimumMinMagMipLinear,
		MinimumAnisotropic,
		MaximumMinMagMipPoint,
		MaximumMinMagPointMipLinear,
		MaximumMinPointMagLinearMipPoint,
		MaximumMinPointMagMipLinear,
		MaximumMinLinearMagMipPoint,
		MaximumMinLinearMagPointMipLinear,
		MaximumMinMagLinearMipPoint,
		MaximumMinMagMipLinear,
		MaximumAnisotropic,
	};

	enum class BufferCreateFlags
	{
		None = 0,
		// Update
		Static = (1 << 0),
		Dynamic = (1 << 1),
		// Usage
		UnorderedAccess = (1 << 2),
		CopySrc = (1 << 3),
		CopyDst = (1 << 4),
		IndirectArguments = (1 << 5),
		ShaderResource = (1 << 6),
		VertexBuffer = (1 << 7),
		IndexBuffer = (1 << 8),
		UniformBuffer = (1 << 9),
		AccelerationStruture = (1 << 10),
		ShaderBindingTable = (1 << 11),
		// Memory access
		CreateMapped = (1 << 12),
		CpuOnly = (1 << 13),
		GpuOnly = (1 << 14),
		CpuToGpu = (1 << 15),
		GpuToCpu = (1 << 16),
	};
	ENUM_CLASS_OPERATORS(BufferCreateFlags);

	enum class RenderBackendResourceState
	{
		Undefined = 0,
		// Read only
		Present = (1 << 0),
		IndirectArgument = (1 << 1),
		VertexBuffer = (1 << 2),
		IndexBuffer = (1 << 3),
		ShaderResource = (1 << 4),
		CopySrc = (1 << 5),
		DepthStencilReadOnly = (1 << 7),
		// Write only
		RenderTarget = (1 << 8),
		CopyDst = (1 << 9),
		// Read-write
		UnorderedAccess = (1 << 10),
		DepthStencil = (1 << 11),

		ReadOnlyMask = Present | IndirectArgument | VertexBuffer | IndexBuffer | ShaderResource | CopySrc | DepthStencilReadOnly,
		WriteOnlyMask = RenderTarget | CopyDst,
		ReadWriteMask = UnorderedAccess | DepthStencil,
		WritableMask = RenderTarget | CopyDst | UnorderedAccess | DepthStencil,
	};
	ENUM_CLASS_OPERATORS(RenderBackendResourceState);

	struct RenderBackendViewport
	{
		float x;
		float y;
		float width;
		float height;
		float minDepth;
		float maxDepth;
		RenderBackendViewport() = default;
		RenderBackendViewport(float width, float height)
			: x(0.0f), y(0.0f), width(width), height(height), minDepth(0.0f), maxDepth(1.0f) {}
		RenderBackendViewport(float x, float y, float width, float height, float minDepth = 0.0f, float maxDepth = 1.0f)
			: x(x), y(y), width(width), height(height), minDepth(minDepth), maxDepth(maxDepth) {}
	};

	struct RenderBackendScissor
	{
		int32 left;
		int32 top;
		uint32 width;
		uint32 height;
		RenderBackendScissor() = default;
		RenderBackendScissor(uint32 width, uint32 height)
			: left(0), top(0), width(width), height(height) {}
		RenderBackendScissor(int32 left, int32 top, uint32 width, uint32 height)
			: left(left), top(top), width(width), height(height) {}
	};

	struct RenderBackendTextureSubresourceRange
	{
		RenderBackendTextureSubresourceRange() = default;
		RenderBackendTextureSubresourceRange(uint32 firstLevel, uint32 mipLevels, uint32 firstLayer, uint32 arrayLayers)
			: firstLevel(firstLevel), mipLevels(mipLevels), firstLayer(firstLayer), arrayLayers(arrayLayers) {}
		FORCEINLINE bool operator==(const RenderBackendTextureSubresourceRange& rhs) const
		{
			return (firstLevel == rhs.firstLevel)
				&& (mipLevels == rhs.mipLevels)
				&& (firstLayer == rhs.firstLayer)
				&& (arrayLayers == rhs.arrayLayers);
		}
		uint32 firstLevel;
		uint32 mipLevels;
		uint32 firstLayer;
		uint32 arrayLayers;
	};

	struct BufferSubresourceRange
	{
		uint64 offset;
		uint64 size;
	};

	struct TextureSubresourceLayers
	{
		uint32 mipLevel;
		uint32 firstLayer;
		uint32 arrayLayers;
	};

	struct RenderTargetClearValue
	{
		static RenderTargetClearValue CreateColorValueFloat4(float r, float g, float b, float a)
		{
			 return RenderTargetClearValue(r, g, b, a);
		}
		static RenderTargetClearValue CreateDepthValue(float depth)
		{
			return RenderTargetClearValue(depth, 0);
		}
		static RenderTargetClearValue CreateDepthStencilValue(float depth, uint32 stencil)
		{
			return RenderTargetClearValue(depth, stencil);
		}
		union
		{
			struct ClearColorValue
			{
				union
				{
					float float32[4];
					int32 int32[4];
					uint32 uint32[4];
				};
			} colorValue;
			struct ClearDepthStencilValue
			{
				float depth;
				uint32 stencil;
			} depthStencilValue;
		};
		RenderTargetClearValue() = default;
		RenderTargetClearValue(float r, float g, float b, float a)
		{
			colorValue.float32[0] = r;
			colorValue.float32[1] = g;
			colorValue.float32[2] = b;
			colorValue.float32[3] = a;
		}
		RenderTargetClearValue(float depth, uint32 stencil)
		{
			depthStencilValue.depth = depth;
			depthStencilValue.stencil = stencil;
		}
	};

	struct RenderStatistics
	{
		uint64 nonIndexedDraws;
		uint64 indexedDraws;
		uint64 nonIndexedIndirectDraws;
		uint64 indexedIndirectDraws;
		uint64 computeDispatches;
		uint64 computeIndirectDispatches;
		uint64 traceRayDispatches;
		uint64 vertices;
		uint64 pipelines;
		uint64 transitions;
		uint64 renderPasses;
		void Add(const RenderStatistics& other)
		{
			nonIndexedDraws += other.nonIndexedDraws;
			indexedDraws += other.indexedDraws;
			nonIndexedIndirectDraws += other.nonIndexedIndirectDraws;
			indexedIndirectDraws += other.indexedIndirectDraws;
			computeDispatches += other.computeDispatches;
			computeIndirectDispatches += other.computeIndirectDispatches;
			traceRayDispatches += other.traceRayDispatches;
			vertices += other.vertices;
			pipelines += other.pipelines;
			transitions += other.transitions;
			renderPasses += other.renderPasses;
		}
	};

	struct RenderBackendBarrier
	{
		enum class ResourceType
		{
			Texture,
			Buffer,
		};
		ResourceType type;
		RenderBackendResourceState srcState;
		RenderBackendResourceState dstState;
		union
		{
			struct
			{
				RenderBackendTextureHandle texture;
				RenderBackendTextureSubresourceRange textureRange;
			};
			struct
			{
				RenderBackendBufferHandle buffer;
				BufferSubresourceRange bufferRange;
			};
		};
		RenderBackendBarrier(RenderBackendTextureHandle texture, RenderBackendTextureSubresourceRange range, RenderBackendResourceState srcState, RenderBackendResourceState dstState)
			: type(ResourceType::Texture), texture(texture), textureRange(range), srcState(srcState), dstState(dstState) {}
		RenderBackendBarrier(RenderBackendBufferHandle buffer, BufferSubresourceRange range, RenderBackendResourceState srcState, RenderBackendResourceState dstState)
			: type(ResourceType::Buffer), buffer(buffer), bufferRange(range), srcState(srcState), dstState(dstState) {}
	};

	struct RenderPassInfo
	{
		struct ColorRenderTarget
		{
			RenderBackendTextureHandle texture;
			uint32 mipLevel;
			uint32 arrayLayer;
			RenderTargetLoadOp loadOp;
			RenderTargetStoreOp storeOp;
		};
		struct DepthStencilRenderTarget
		{
			RenderBackendTextureHandle texture;
			uint32 mipLevel;
			uint32 arrayLayer;
			RenderTargetLoadOp depthLoadOp;
			RenderTargetStoreOp depthStoreOp;
			RenderTargetLoadOp stencilLoadOp;
			RenderTargetStoreOp stencilStoreOp;
		};
		ColorRenderTarget colorRenderTargets[MaxNumSimultaneousColorRenderTargets];
		DepthStencilRenderTarget depthStencilRenderTarget;
	};

	struct ShaderBindingTable
	{
		RenderBackendBufferHandle buffer;
		uint64 offset;
		uint64 size;
		uint64 stride;
	};

	struct RenderBackendBufferDesc
	{
		static RenderBackendBufferDesc Create(uint32 elementSize, uint32 elementCount, BufferCreateFlags flags)
		{
			return RenderBackendBufferDesc(elementSize, elementCount, flags);
		}
		static RenderBackendBufferDesc CreateIndirect(uint32 elementSize, uint32 elementCount)
		{
			auto flags = BufferCreateFlags::Static | BufferCreateFlags::IndirectArguments | BufferCreateFlags::UnorderedAccess | BufferCreateFlags::ShaderResource;
			return RenderBackendBufferDesc(elementSize, elementCount, flags);
		}
		static RenderBackendBufferDesc CreateByteAddress(uint64 bytes)
		{
			auto flags = BufferCreateFlags::UnorderedAccess | BufferCreateFlags::ShaderResource | BufferCreateFlags::IndexBuffer;
			return RenderBackendBufferDesc(4, (uint32)(bytes >> 2), flags);
		}
		static RenderBackendBufferDesc CreateStructured(uint32 elementSize, uint32 elementCount)
		{
			auto flags = BufferCreateFlags::Static | BufferCreateFlags::UnorderedAccess | BufferCreateFlags::ShaderResource;
			return RenderBackendBufferDesc(elementSize, elementCount, flags);
		}
		static RenderBackendBufferDesc CreateShaderBindingTable(uint64 bytes)
		{
			auto flags = BufferCreateFlags::ShaderBindingTable | BufferCreateFlags::CpuOnly | BufferCreateFlags::CreateMapped;
			return RenderBackendBufferDesc(4, (uint32)(bytes >> 2), flags);
		}
		static RenderBackendBufferDesc CreateShaderBindingTable(uint64 handleSize, uint32 handleCount)
		{
			auto flags = BufferCreateFlags::ShaderBindingTable | BufferCreateFlags::CpuOnly | BufferCreateFlags::CreateMapped;
			return RenderBackendBufferDesc(4, (uint32)((handleSize * handleCount) >> 2), flags);
		}
		RenderBackendBufferDesc() = default;
		RenderBackendBufferDesc(uint32 elementSize, uint32 elementCount, BufferCreateFlags flags)
			: elementSize(elementSize)
			, elementCount(elementCount)
			, size(elementSize* elementCount)
			, flags(flags) {}

		uint64 size;
		uint32 elementSize;
		uint32 elementCount;
		BufferCreateFlags flags;
	};

	struct RenderBackendTextureDesc
	{
		static RenderBackendTextureDesc CreateTexture2D(
			uint32 width,
			uint32 height,
			uint32 mipLevels,
			PixelFormat format)
		{
			TextureCreateFlags flags = TextureCreateFlags::ShaderResource;
			return RenderBackendTextureDesc(width, height, 1, mipLevels, 1, 1, TextureType::Texture2D, format, flags, {});
		}

		static RenderBackendTextureDesc Create2D(
			uint32 width,
			uint32 height,
			PixelFormat format,
			TextureCreateFlags flags,
			RenderTargetClearValue clearValue = {},
			uint32 mipLevels = 1,
			uint32 samples = 1)
		{
			return RenderBackendTextureDesc(width, height, 1, mipLevels, 1, samples, TextureType::Texture2D, format, flags, clearValue);
		}

		static RenderBackendTextureDesc Create2DArray(
			uint32 width,
			uint32 height,
			PixelFormat format,
			TextureCreateFlags flags,
			uint32 arraySize,
			RenderTargetClearValue clearValue = {},
			uint32 mipLevels = 1,
			uint32 samples = 1)
		{
			return RenderBackendTextureDesc(width, height, 1, mipLevels, arraySize, samples, TextureType::Texture2D, format, flags, clearValue);
		}

		static RenderBackendTextureDesc Create3D(
			uint32 width,
			uint32 height,
			uint32 depth,
			PixelFormat format,
			TextureCreateFlags flags,
			uint32 mipLevels = 1,
			uint32 samples = 1)
		{
			return RenderBackendTextureDesc(width, height, depth, mipLevels, 1, samples, TextureType::Texture3D, format, flags, {});
		}

		static RenderBackendTextureDesc CreateCube(
			uint32 sizeInPixels,
			PixelFormat format,
			TextureCreateFlags flags,
			uint32 mipLevels = 1,
			uint32 samples = 1)
		{
			return RenderBackendTextureDesc(sizeInPixels, sizeInPixels, 1, mipLevels, 6, samples, TextureType::TextureCube, format, flags, {});
		}

		RenderBackendTextureDesc() = default;
		RenderBackendTextureDesc(
			uint32 width,
			uint32 height,
			uint32 depth,
			uint32 mipLevels,
			uint32 arraySize,
			uint32 samples,
			TextureType type,
			PixelFormat format,
			TextureCreateFlags flags,
			RenderTargetClearValue clearValue)
			: width(width)
			, height(height)
			, depth(depth)
			, mipLevels(mipLevels)
			, arrayLayers(arraySize)
			, samples(samples)
			, type(type)
			, format(format)
			, flags(flags)
			, clearValue(clearValue) {}

		bool operator==(const RenderBackendTextureDesc& rhs) const
		{
			return width == rhs.width
				&& height == height
				&& depth == rhs.depth
				&& mipLevels == rhs.mipLevels
				&& arrayLayers == rhs.arrayLayers
				&& samples == rhs.samples
				&& type == rhs.type
				&& format == rhs.format
				&& flags == rhs.flags;
		}

		uint32 width;
		uint32 height;
		uint32 depth;
		uint32 mipLevels;
		uint32 arrayLayers;
		uint32 samples;
		TextureType type;
		PixelFormat format;
		TextureCreateFlags flags;
		RenderTargetClearValue clearValue;
	};

	struct RenderBackendTextureSRVDesc
	{
		static RenderBackendTextureSRVDesc Create(RenderBackendTextureHandle texture)
		{
			return RenderBackendTextureSRVDesc(texture, 0, REMAINING_MIP_LEVELS, 0, REMAINING_ARRAY_LAYERS);
		}
		static RenderBackendTextureSRVDesc Create(RenderBackendTextureHandle texture, uint32 baseMipLevel, uint32 mipLevelCount, uint32 baseArrayLayer, uint32 arrayLayerCount)
		{
			return RenderBackendTextureSRVDesc(texture, baseMipLevel, mipLevelCount, baseArrayLayer, arrayLayerCount);
		}
		static RenderBackendTextureSRVDesc CreateForMipLevel(RenderBackendTextureHandle texture, uint32 mipLevel, uint32 baseArrayLayer = 0, uint32 arrayLayerCount = 1)
		{
			return RenderBackendTextureSRVDesc(texture, mipLevel, 1, baseArrayLayer, arrayLayerCount);
		}
		RenderBackendTextureSRVDesc() {}
		RenderBackendTextureSRVDesc(RenderBackendTextureHandle texture, uint32 baseMipLevel, uint32 numMipLevels, uint32 baseArrayLayer, uint32 numArrayLayers)
			: texture(texture)
			, baseMipLevel(baseMipLevel)
			, numMipLevels(numMipLevels)
			, baseArrayLayer(baseArrayLayer)
			, numArrayLayers(numArrayLayers) {}
		RenderBackendTextureHandle texture;
		uint32 baseMipLevel;
		uint32 numMipLevels;
		uint32 baseArrayLayer;
		uint32 numArrayLayers;
	};

	struct RenderBackendTextureUAVDesc
	{
		static RenderBackendTextureUAVDesc Create(RenderBackendTextureHandle texture, uint32 mipLevel = 0)
		{
			return RenderBackendTextureUAVDesc(texture, mipLevel);
		}
		RenderBackendTextureUAVDesc() = default;
		RenderBackendTextureUAVDesc(RenderBackendTextureHandle texture, uint32 mipLevel)
			: texture(texture), mipLevel(mipLevel) {}
		RenderBackendTextureHandle texture = RenderBackendTextureHandle::NullHandle;
		uint32 mipLevel = 0;
	};

	struct RenderBackendSamplerDesc
	{
		static RenderBackendSamplerDesc CreateLinearClamp(float mipLodBias, float minLod, float maxLod, uint32 maxAnisotropy)
		{
			return RenderBackendSamplerDesc(
				Filter::MinMagMipLinear,
				TextureAddressMode::Clamp,
				TextureAddressMode::Clamp,
				TextureAddressMode::Clamp,
				mipLodBias,
				minLod,
				maxLod,
				maxAnisotropy,
				CompareOp::Never);
		}
		static RenderBackendSamplerDesc CreateLinearWarp(float mipLodBias, float minLod, float maxLod, uint32 maxAnisotropy)
		{
			return RenderBackendSamplerDesc(
				Filter::MinMagMipLinear,
				TextureAddressMode::Warp,
				TextureAddressMode::Warp,
				TextureAddressMode::Warp,
				mipLodBias,
				minLod,
				maxLod,
				maxAnisotropy,
				CompareOp::Never);
		}
		static RenderBackendSamplerDesc CreateComparisonLinearClamp(float mipLodBias, float minLod, float maxLod, uint32 maxAnisotropy, CompareOp compareOp)
		{
			return RenderBackendSamplerDesc(
				Filter::ComparisonMinMagMipLinear,
				TextureAddressMode::Clamp,
				TextureAddressMode::Clamp,
				TextureAddressMode::Clamp,
				mipLodBias,
				minLod,
				maxLod,
				maxAnisotropy,
				compareOp);
		}
		RenderBackendSamplerDesc(
			Filter filter,
			TextureAddressMode addressModeU,
			TextureAddressMode addressModeV,
			TextureAddressMode addressModeW,
			float mipLodBias,
			float minLod,
			float maxLod,
			uint32 maxAnisotropy,
			CompareOp compareOp)
			: filter(filter)
			, addressModeU(addressModeU)
			, addressModeV(addressModeV)
			, addressModeW(addressModeW)
			, mipLodBias(mipLodBias)
			, minLod(minLod)
			, maxLod(maxLod)
			, maxAnisotropy(maxAnisotropy) 
			, compareOp(compareOp) {}
		Filter filter;
		TextureAddressMode addressModeU;
		TextureAddressMode addressModeV;
		TextureAddressMode addressModeW;
		float mipLodBias;
		float minLod;
		float maxLod;
		uint32 maxAnisotropy;
		CompareOp compareOp;
	};

	struct RenderBackendShaderDesc
	{
		RasterizationState rasterizationState;
		DepthStencilState depthStencilState;
		ColorBlendState colorBlendState;
		std::string entryPoints[(uint32)RenderBackendShaderStage::Count] = {};
		ShaderBlob stages[(uint32)RenderBackendShaderStage::Count] = {};
	};

	struct ShaderArguments
	{
		struct TextureSRV
		{
			uint32 slot;
			RenderBackendTextureSRVDesc srv;
		};

		struct TextureUAV
		{
			uint32 slot;
			RenderBackendTextureUAVDesc uav;
		};

		struct Buffer
		{
			uint32 slot;
			RenderBackendBufferHandle handle;
			uint32 offset;
		};

		struct AS
		{
			uint32 slot;
			RenderBackendRayTracingAccelerationStructureHandle handle;
		};

		struct Slot
		{
			int32 type = 0;
			union
			{
				TextureSRV srvSlot;
				TextureUAV uavSlot;
				Buffer bufferSlot;
				AS asSlot;
			};
		};

		void BindTextureSRV(uint32 slot, const RenderBackendTextureSRVDesc& srv)
		{
			slots[slot] = { .type = 1, .srvSlot = { slot, srv } };
		}

		void BindTextureUAV(uint32 slot, const RenderBackendTextureUAVDesc& uav)
		{
			slots[slot] = { .type = 2, .uavSlot = { slot, uav } };
		}

		void BindBuffer(uint32 slot, RenderBackendBufferHandle buffer, uint32 offset)
		{
			slots[slot] = { .type = 3, .bufferSlot = { slot, buffer, offset } };
		}

		void BindAS(uint32 slot, RenderBackendRayTracingAccelerationStructureHandle as)
		{
			slots[slot] = { .type = 4, .asSlot = { slot, as } };
		}

		void PushConstants(uint32 slot, float value)
		{
			data[slot] = value;
		}

		Slot slots[16];
		float data[16];
	};

	enum RenderBackendAccelerationStructureBuildFlags
	{
		None = 0,
		AllowUpdate = 1 << 0,
		AllowCompaction = 1 << 1,
		PreferFastTrace = 1 << 2,
		PreferFastBuild = 1 << 3,
		MinimizeMemory = 1 << 4,
	};

	enum class RenderBackendRayTracingInstanceFlags
	{
		None = 0,
		TriangleFacingCullDisable = 1 << 0,
		TriangleFrontCounterclockwise = 1 << 1,
		ForceOpaque = 1 << 2,
		ForceNoOpaque = 1 << 3,
	};

	struct RenderBackendRayTracingInstance
	{
		Matrix4x4 transformMatrix;
		uint32 instanceID;
		uint32 instanceMask;
		uint32 instanceContributionToHitGroupIndex;
		RenderBackendRayTracingInstanceFlags flags;
		RenderBackendRayTracingAccelerationStructureHandle blas;
	};

	enum class RenderBackendGeometryType
	{
		Triangles = 0,
		AABBs = 1,
	};

	enum class RenderBackendGeometryFlags
	{
		None = 0,
		Opaque = 1 << 0,
		NoDuplicateAnyHitInvocation = 1 << 1,
	};

	struct RenderBackendGeometryTriangleDesc
	{
		uint32 numIndices;
		uint32 numVertices;
		uint32 vertexStride;
		RenderBackendBufferHandle vertexBuffer;
		uint32 vertexOffset;
		RenderBackendBufferHandle indexBuffer;
		uint32 indexOffset;
		RenderBackendBufferHandle transformBuffer;
		uint32 transformOffset;
	};

	struct RenderBackendGeometryAABBDesc
	{
		RenderBackendBufferHandle buffer;
		uint32 offset;
	};

	struct RenderBackendGeometryDesc
	{
		RenderBackendGeometryType type;
		RenderBackendGeometryFlags flags;
		//union
		//{
		RenderBackendGeometryTriangleDesc triangleDesc;
		RenderBackendGeometryAABBDesc aabbDesc;
		//};
	};

	struct RenderBackendBottomLevelASDesc
	{
		RenderBackendAccelerationStructureBuildFlags buildFlags;
		uint32 numGeometries;
		RenderBackendGeometryDesc* geometryDescs;
	};

	struct RenderBackendTopLevelASDesc
	{
		RenderBackendAccelerationStructureBuildFlags buildFlags;
		RenderBackendGeometryFlags geometryFlags;
		uint32 numInstances;
		RenderBackendRayTracingInstance* instances;
	};

	struct RenderBackendAccelerationStructureRange
	{
		uint32 numPrimitives;
		uint32 primitiveOffset;
		uint32 firstVertex;
		uint32 transformOffset;
	};

	enum class RenderBackendRayTracingShaderGroupType
	{
		RayGen,
		Miss,
		TrianglesHitGroup,
		ProceduralHitGroup,
	};

	struct RenderBackendRayTracingShaderDesc
	{
		RenderBackendShaderStage stage;
		std::string entry;
		ShaderBlob code;
	};

	struct RenderBackendRayTracingShaderGroupDesc
	{
		static const uint32 ShaderUnused = ~0u;

		static RenderBackendRayTracingShaderGroupDesc CreateRayGen(uint32 rayGenerationShader)
		{
			return RenderBackendRayTracingShaderGroupDesc(
				RenderBackendRayTracingShaderGroupType::RayGen,
				rayGenerationShader,
				ShaderUnused,
				ShaderUnused,
				ShaderUnused,
				ShaderUnused);
		}

		static RenderBackendRayTracingShaderGroupDesc CreateMiss(uint32 missShader)
		{
			return RenderBackendRayTracingShaderGroupDesc(
				RenderBackendRayTracingShaderGroupType::Miss,
				ShaderUnused,
				missShader,
				ShaderUnused,
				ShaderUnused,
				ShaderUnused);
		}

		static RenderBackendRayTracingShaderGroupDesc CreateTrianglesHitGroup(uint32 closestHitShader, uint32 anyHitShader, uint32 intersectionShader)
		{
			return RenderBackendRayTracingShaderGroupDesc(
				RenderBackendRayTracingShaderGroupType::TrianglesHitGroup,
				ShaderUnused,
				ShaderUnused,
				closestHitShader,
				anyHitShader,
				intersectionShader);
		}

		RenderBackendRayTracingShaderGroupDesc()
			: type(RenderBackendRayTracingShaderGroupType::RayGen)
			, rayGenerationShader(ShaderUnused)
			, missShader(ShaderUnused)
			, closestHitShader(ShaderUnused)
			, anyHitShader(ShaderUnused)
			, intersectionShader(ShaderUnused)
		{

		}

		RenderBackendRayTracingShaderGroupDesc(
			RenderBackendRayTracingShaderGroupType type,
			uint32 rayGenerationShader,
			uint32 missShader,
			uint32 closestHitShader,
			uint32 anyHitShader,
			uint32 intersectionShader)
			: type(type)
			, rayGenerationShader(rayGenerationShader)
			, missShader(missShader)
			, closestHitShader(closestHitShader)
			, anyHitShader(anyHitShader)
			, intersectionShader(intersectionShader)
		{

		}

		RenderBackendRayTracingShaderGroupType type;
		uint32 rayGenerationShader;
		uint32 missShader;
		uint32 closestHitShader;
		uint32 anyHitShader;
		uint32 intersectionShader;
	};

	struct RenderBackendRayTracingPipelineStateDesc
	{
		uint32 maxRayRecursionDepth;
		std::vector<RenderBackendRayTracingShaderDesc> shaders;
		std::vector<RenderBackendRayTracingShaderGroupDesc> shaderGroupDescs;
	};

	struct RenderBackendRayTracingShaderBindingTableDesc
	{
		RenderBackendRayTracingPipelineStateHandle rayTracingPipelineState;
		uint32 numShaderRecords;
		std::vector<uint32> shaderGroupIndices;
		std::vector<uint32> shaderRecordSizes;
		std::vector<void*>  shaderRecordValues;
	};

	struct RenderBackendTimingQueryHeapDesc
	{
		uint32 maxRegions;
		RenderBackendTimingQueryHeapDesc() = default;
		RenderBackendTimingQueryHeapDesc(uint32 maxRegions) : maxRegions(maxRegions) {}
	};

	struct RenderBackendOcclusionQueryHeapDesc
	{
		uint32 maxQueries;
		RenderBackendOcclusionQueryHeapDesc() = default;
		RenderBackendOcclusionQueryHeapDesc(uint32 maxQueries) : maxQueries(maxQueries) {}
	};

	//struct GpuProfiler
	//{
	//	// double buffered
	//	static const uint32_t NUM_OF_FRAMES = 3;
	//	static const uint32_t MAX_TIMERS = 512;
	//
	//	Renderer* pRenderer = {};
	//	Buffer* pReadbackBuffer[NUM_OF_FRAMES] = {};
	//	QueryPool* pQueryPool[NUM_OF_FRAMES] = {};
	//	uint64_t* pTimeStamp = NULL;
	//	double                mGpuTimeStampFrequency = 0.0;
	//
	//	uint32_t mProfilerIndex = 0;
	//	uint32_t mBufferIndex = 0;
	//	uint32_t mCurrentTimerCount = 0;
	//	uint32_t mMaxTimerCount = 0;
	//	uint32_t mCurrentPoolIndex = 0;
	//
	//	GpuTimer* pGpuTimerPool = NULL;
	//	GpuTimer* pCurrentNode = NULL;
	//
	//	// MicroProfile
	//	char mGroupName[256] = "GPU";
	//	ProfileThreadLog* pLog = nullptr;
	//
	//	bool mReset = true;
	//	bool mUpdate = false;
	//};


	enum class RenderCommandQueueType : uint8
	{
		None = 0,
		Copy = (1 << 0),
		Compute = (1 << 1),
		Graphics = (1 << 2),
		All = Copy | Compute | Graphics,
	};

	enum class RenderCommandType
	{
		CopyBuffer,
		CopyTexture,
		UpdateBuffer,
		UpdateTexture,
		Barriers,
		Transitions,
		BeginTiming,
		EndTiming,
		ResetTimingQueryHeap,
		Dispatch,
		DispatchIndirect,
		UpdateBottomLevelAS,
		UpdateTopLevelAS,
		TraceRays,
		SetViewport,
		SetScissor,
		BeginRenderPass,
		EndRenderPass,
		Draw,
		DrawIndirect,
		Count,
	};

	template<RenderCommandType commandType, RenderCommandQueueType queueType>
	struct RenderCommand
	{
		static const RenderCommandType Type = commandType;
		static const RenderCommandQueueType QueueType = queueType;
	};

	struct RenderCommandCopyBuffer : RenderCommand<RenderCommandType::CopyBuffer, RenderCommandQueueType::All>
	{
		RenderBackendBufferHandle srcBuffer;
		uint64 srcOffset;
		RenderBackendBufferHandle dstBuffer;
		uint64 dstOffset;
		uint64 bytes;
	};

	struct RenderCommandCopyTexture : RenderCommand<RenderCommandType::CopyTexture, RenderCommandQueueType::All>
	{
		RenderBackendTextureHandle srcTexture;
		Offset3D srcOffset;
		TextureSubresourceLayers srcSubresourceLayers;
		RenderBackendTextureHandle dstTexture;
		Offset3D dstOffset;
		TextureSubresourceLayers dstSubresourceLayers;
		Extent3D extent;
	};

	struct RenderCommandUpdateBuffer : RenderCommand<RenderCommandType::UpdateBuffer, RenderCommandQueueType::All>
	{
		RenderBackendBufferHandle buffer;
		uint64 offset;
		const void* data;
		uint64 size;
	};

	struct RenderCommandUpdateTexture : RenderCommand<RenderCommandType::UpdateTexture, RenderCommandQueueType::All>
	{
		RenderBackendTextureHandle texture;
	};

	struct RenderCommandBarriers : RenderCommand<RenderCommandType::Barriers, RenderCommandQueueType::All>
	{
		uint32 numBarriers;
	};

	struct RenderCommandTransitions : RenderCommand<RenderCommandType::Transitions, RenderCommandQueueType::All>
	{
		uint32 numTransitions;
		RenderBackendBarrier* transitions;
	};

	struct RenderCommandBeginTimingQuery : RenderCommand<RenderCommandType::BeginTiming, RenderCommandQueueType::All>
	{
		RenderBackendTimingQueryHeapHandle timingQueryHeap;
		uint32 region;
	};

	struct RenderCommandEndTimingQuery : RenderCommand<RenderCommandType::EndTiming, RenderCommandQueueType::All>
	{
		RenderBackendTimingQueryHeapHandle timingQueryHeap;
		uint32 region;
	};

	struct RenderCommandResetTimingQueryHeap : RenderCommand<RenderCommandType::ResetTimingQueryHeap, RenderCommandQueueType::Graphics>
	{
		RenderBackendTimingQueryHeapHandle timingQueryHeap;
		uint32 regionStart;
		uint32 regionCount;
	};

	struct RenderCommandDispatch : RenderCommand<RenderCommandType::Dispatch, RenderCommandQueueType::Compute>
	{
		RenderBackendShaderHandle shader;
		ShaderArguments shaderArguments;
		uint32 threadGroupCountX;
		uint32 threadGroupCountY;
		uint32 threadGroupCountZ;
	};

	struct RenderCommandDispatchIndirect : RenderCommand<RenderCommandType::DispatchIndirect, RenderCommandQueueType::Compute>
	{
		RenderBackendShaderHandle shader;
		ShaderArguments shaderArguments;
		RenderBackendBufferHandle argumentBuffer;
		uint64 argumentOffset;
	};

	struct RenderCommandTraceRays : RenderCommand<RenderCommandType::TraceRays, RenderCommandQueueType::Graphics>
	{
		RenderBackendRayTracingPipelineStateHandle pipelineState;
		RenderBackendBufferHandle shaderBindingTable;
		ShaderArguments shaderArguments;
		uint32 width;
		uint32 height;
		uint32 depth;
	};

	struct RenderCommandUpdateBottomLevelAS : RenderCommand<RenderCommandType::UpdateBottomLevelAS, RenderCommandQueueType::Compute>
	{
		RenderBackendRayTracingAccelerationStructureHandle srcBLAS;
		RenderBackendRayTracingAccelerationStructureHandle dstBLAS;
	};

	struct RenderCommandUpdateTopLevelAS : RenderCommand<RenderCommandType::UpdateTopLevelAS, RenderCommandQueueType::Compute>
	{
		RenderBackendRayTracingAccelerationStructureHandle srcTLAS;
		RenderBackendRayTracingAccelerationStructureHandle dstTLAS;
	};

	struct RenderCommandSetViewport : RenderCommand<RenderCommandType::SetViewport, RenderCommandQueueType::Graphics>
	{
		uint32 numViewports;
		RenderBackendViewport viewports[MaxNumViewports];
	};

	struct RenderCommandSetScissor : RenderCommand<RenderCommandType::SetScissor, RenderCommandQueueType::Graphics>
	{
		uint32 numScissors;
		RenderBackendScissor scissors[MaxNumViewports];
	};

	struct RenderCommandBeginRenderPass : RenderCommand<RenderCommandType::BeginRenderPass, RenderCommandQueueType::Graphics>
	{
		RenderPassInfo renderPassInfo;
	};

	struct RenderCommandEndRenderPass : RenderCommand<RenderCommandType::EndRenderPass, RenderCommandQueueType::Graphics>
	{

	};

	struct RenderCommandDraw : RenderCommand<RenderCommandType::Draw, RenderCommandQueueType::Graphics>
	{
		RenderBackendShaderHandle shader;
		ShaderArguments shaderArguments;
		RenderBackendBufferHandle indexBuffer;
		union
		{
			struct
			{
				uint32 numVertices;
				uint32 numInstances;
				uint32 firstVertex;
				uint32 firstInstance;
			};
			struct
			{
				uint32 numIndices;
				uint32 numInstances;
				uint32 firstIndex;
				int32 vertexOffset;
				uint32 firstInstance;
			};
		};
		PrimitiveTopology topology;
	};

	struct RenderCommandDrawIndirect : RenderCommand<RenderCommandType::DrawIndirect, RenderCommandQueueType::Graphics>
	{
		RenderBackendShaderHandle shader;
		ShaderArguments shaderArguments;
		RenderBackendBufferHandle indexBuffer;
		RenderBackendBufferHandle argumentBuffer;
		uint64 offset;
		uint32 numDraws;
		uint32 stride;
		PrimitiveTopology topology;
	};

	struct RenderBackendGpuMask
	{
	public:
		FORCEINLINE static const RenderBackendGpuMask All()
		{
			return RenderBackendGpuMask(~0u);
		}

		FORCEINLINE RenderBackendGpuMask() : RenderBackendGpuMask(RenderBackendGpuMask::All()) {}

		FORCEINLINE bool operator ==(const RenderBackendGpuMask& rhs) const { return mask == rhs.mask; }
		FORCEINLINE bool operator !=(const RenderBackendGpuMask& rhs) const { return mask != rhs.mask; }
		void operator |=(const RenderBackendGpuMask& rhs) { mask |= rhs.mask; }
		void operator &=(const RenderBackendGpuMask& rhs) { mask &= rhs.mask; }
		FORCEINLINE RenderBackendGpuMask operator &(const RenderBackendGpuMask& rhs) const
		{
			return RenderBackendGpuMask(mask & rhs.mask);
		}
		FORCEINLINE RenderBackendGpuMask operator |(const RenderBackendGpuMask& rhs) const
		{
			return RenderBackendGpuMask(mask | rhs.mask);
		}

		struct Iterator
		{
			static uint32 CountTrailingZeros(uint32 value)
			{
				if (value == 0)
				{
					return 32;
				}
				unsigned long result;
				_BitScanForward(&result, value);
				return (uint32)result;
			}
			FORCEINLINE explicit Iterator(uint32 mask) : mask(mask), firstNonZeroBit(0)
			{
				firstNonZeroBit = CountTrailingZeros(mask);
			}

			FORCEINLINE explicit Iterator(const RenderBackendGpuMask& gpuMask) : Iterator(gpuMask.mask)
			{
			}

			FORCEINLINE Iterator& operator++()
			{
				mask &= ~(1 << firstNonZeroBit);
				firstNonZeroBit = CountTrailingZeros(mask);
				return *this;
			}

			FORCEINLINE Iterator operator++(int)
			{
				Iterator copy(*this);
				++* this;
				return copy;
			}

			FORCEINLINE uint32 operator*() const { return firstNonZeroBit; }
			FORCEINLINE bool operator !=(const Iterator& rhs) const { return mask != rhs.mask; }
			FORCEINLINE explicit operator bool() const { return mask != 0; }
			FORCEINLINE bool operator !() const { return !(bool)*this; }

		private:
			uint32 mask;
			uint32 firstNonZeroBit;
		};

		FORCEINLINE friend RenderBackendGpuMask::Iterator begin(const RenderBackendGpuMask& gpuMask) { return RenderBackendGpuMask::Iterator(gpuMask.mask); }
		FORCEINLINE friend RenderBackendGpuMask::Iterator end(const RenderBackendGpuMask& gpuMask) { return RenderBackendGpuMask::Iterator(0); }

	private:
		FORCEINLINE explicit RenderBackendGpuMask(uint32 mask) : mask(mask) {}
		uint32 mask;
	};

	using PhysicalDeviceID = uint32;

	class RenderCommandList;

	struct RenderBackend
	{
		void* instance;
		void (*Tick)(void* instance);
		void (*CreateRenderDevices)(void* instance, PhysicalDeviceID* physicalDeviceIDs, uint32 numDevices, uint32* outDeviceMasks);
		void (*DestroyRenderDevices)(void* instance);
		void (*FlushRenderDevices)(void* instance);
		RenderBackendSwapChainHandle (*CreateSwapChain)(void* instance, uint32 deviceMask, uint64 windowHandle);
		void (*DestroySwapChain)(void* instance, RenderBackendSwapChainHandle swapChain);
		void (*ResizeSwapChain)(void* instance, RenderBackendSwapChainHandle swapChain, uint32* width, uint32* height);
		bool (*PresentSwapChain)(void* instance, RenderBackendSwapChainHandle swapChain);
		RenderBackendTextureHandle (*GetActiveSwapChainBuffer)(void* instance, RenderBackendSwapChainHandle swapChain);
		RenderBackendBufferHandle (*CreateBuffer)(void* instance, uint32 deviceMask, const RenderBackendBufferDesc* desc, const char* name);
		void (*ResizeBuffer)(void* instance, RenderBackendBufferHandle buffer, uint64 size);
		void (*MapBuffer)(void* instance, RenderBackendBufferHandle buffer, void** data);
		void (*UnmapBuffer)(void* instance, RenderBackendBufferHandle buffer);
		void (*DestroyBuffer)(void* instance, RenderBackendBufferHandle buffer);
		RenderBackendTextureHandle (*CreateTexture)(void* instance, uint32 deviceMask, const RenderBackendTextureDesc* desc, const void* data, const char* name);
		void (*DestroyTexture)(void* instance, RenderBackendTextureHandle texture);
		RenderBackendTextureSRVHandle (*CreateTextureSRV)(void* instance, uint32 deviceMask, const RenderBackendTextureSRVDesc* desc, const char* name);
		int32(*GetTextureSRVDescriptorIndex)(void* instance, uint32 deviceMask, RenderBackendTextureHandle srv);
		RenderBackendTextureUAVHandle (*CreateTextureUAV)(void* instance, uint32 deviceMask, const RenderBackendTextureUAVDesc* desc, const char* name);
		int32 (*GetTextureUAVDescriptorIndex)(void* instance, uint32 deviceMask, RenderBackendTextureHandle uav);
		RenderBackendSamplerHandle (*CreateSampler)(void* instance, uint32 deviceMask, const RenderBackendSamplerDesc* desc, const char* name);
		void (*DestroySampler)(void* instance, RenderBackendSamplerHandle sampler);
		RenderBackendShaderHandle (*CreateShader)(void* instance, uint32 deviceMask, const RenderBackendShaderDesc* desc, const char* name);
		void (*DestroyShader)(void* instance, RenderBackendShaderHandle shader);
		RenderBackendTimingQueryHeapHandle (*CreateTimingQueryHeap)(void* instance, uint32 deviceMask, const RenderBackendTimingQueryHeapDesc* desc, const char* name);
		void (*DestroyTimingQueryHeap)(void* instance, RenderBackendTimingQueryHeapHandle timingQueryHeap);
		bool (*GetTimingQueryHeapResults)(void* instance, RenderBackendTimingQueryHeapHandle timingQueryHeap, uint32 regionStart, uint32 regionCount, void* results);
		RenderBackendOcclusionQueryHeapHandle (*CreateOcclusionQueryHeap)(void* instance, uint32 deviceMask, const RenderBackendOcclusionQueryHeapDesc* desc, const char* name);
		void (*DestroyOcclusionQueryHeap)(void* instance, RenderBackendOcclusionQueryHeapHandle occlusionQueryHeap);
		void (*SubmitRenderCommandLists)(void* instance, RenderCommandList** commandLists, uint32 numCommandLists);
		void (*GetRenderStatistics)(void* instance, uint32 deviceMask, RenderStatistics* statistics);
		RenderBackendRayTracingAccelerationStructureHandle (*CreateBottomLevelAS)(void* instance, uint32 deviceMask, const RenderBackendBottomLevelASDesc* desc, const char* name);
		RenderBackendRayTracingAccelerationStructureHandle (*CreateTopLevelAS)(void* instance, uint32 deviceMask, const RenderBackendTopLevelASDesc* desc, const char* name);
		RenderBackendRayTracingPipelineStateHandle (*CreateRayTracingPipelineState)(void* instance, uint32 deviceMask, const RenderBackendRayTracingPipelineStateDesc* desc, const char* name);
		RenderBackendBufferHandle (*CreateRayTracingShaderBindingTable)(void* instance, uint32 deviceMask, const RenderBackendRayTracingShaderBindingTableDesc* desc, const char* name);
	};

	void RenderBackendTick(RenderBackend* backend);
	void RenderBackendCreateRenderDevices(RenderBackend* backend, PhysicalDeviceID* physicalDeviceIDs, uint32 numDevices, uint32* outDeviceMasks);
	void RenderBackendDestroyRenderDevices(RenderBackend* backend);
	void RenderBackendFlushRenderDevices(RenderBackend* backend);
	RenderBackendSwapChainHandle RenderBackendCreateSwapChain(RenderBackend* backend, uint32 deviceMask, uint64 windowHandle);
	void RenderBackendDestroySwapChain(RenderBackend* backend, RenderBackendSwapChainHandle swapChain);
	void RenderBackendResizeSwapChain(RenderBackend* backend, RenderBackendSwapChainHandle swapChain, uint32* width, uint32* height);
	bool RenderBackendPresentSwapChain(RenderBackend* backend, RenderBackendSwapChainHandle swapChain);
	RenderBackendTextureHandle RenderBackendGetActiveSwapChainBuffer(RenderBackend* backend, RenderBackendSwapChainHandle swapChain);
	RenderBackendBufferHandle RenderBackendCreateBuffer(RenderBackend* backend, uint32 deviceMask, const RenderBackendBufferDesc* desc, const char* name);
	void RenderBackendResizeBuffer(RenderBackend* backend, RenderBackendBufferHandle buffer, uint64 size);
	void RenderBackendMapBuffer(RenderBackend* backend, RenderBackendBufferHandle buffer, void** data);
	void RenderBackendUnmapBuffer(RenderBackend* backend, RenderBackendBufferHandle buffer);
	void RenderBackendDestroyBuffer(RenderBackend* backend, RenderBackendBufferHandle buffer);
	RenderBackendTextureHandle RenderBackendCreateTexture(RenderBackend* backend, uint32 deviceMask, const RenderBackendTextureDesc* desc, const void* data, const char* name);
	void RenderBackendDestroyTexture(RenderBackend* backend, RenderBackendTextureHandle texture);
	RenderBackendTextureSRVHandle RenderBakendCreateTextureSRV(RenderBackend* backend, uint32 deviceMask, const RenderBackendTextureSRVDesc* desc, const char* name);
	int32 RenderBackendGetTextureSRVDescriptorIndex(RenderBackend* backend, uint32 deviceMask, RenderBackendTextureHandle srv);
	RenderBackendTextureUAVHandle RenderBackendCreateTextureUAV(RenderBackend* backend, uint32 deviceMask, const RenderBackendTextureUAVDesc* desc, const char* name);
	int32 RenderBackendGetTextureUAVDescriptorIndex(RenderBackend* backend, uint32 deviceMask, RenderBackendTextureHandle uav);
	RenderBackendSamplerHandle RenderBackendCreateSampler(RenderBackend* backend, uint32 deviceMask, const RenderBackendSamplerDesc* desc, const char* name);
	void RenderBackendDestroySampler(RenderBackend* backend, RenderBackendSamplerHandle sampler);
	RenderBackendShaderHandle RenderBackendCreateShader(RenderBackend* backend, uint32 deviceMask, const RenderBackendShaderDesc* desc, const char* name);
	void RenderBackendDestroyShader(RenderBackend* backend, RenderBackendShaderHandle shader);
	RenderBackendTimingQueryHeapHandle RenderBackendCreateTimingQueryHeap(RenderBackend* backend, uint32 deviceMask, const RenderBackendTimingQueryHeapDesc* desc, const char* name);
	void RenderBackendDestroyTimingQueryHeap(RenderBackend* backend, RenderBackendTimingQueryHeapHandle timingQueryHeap);
	bool RenderBackendGetTimingQueryHeapResults(RenderBackend* backend, RenderBackendTimingQueryHeapHandle timingQueryHeap, uint32 regionStart, uint32 regionCount, void* results);
	RenderBackendOcclusionQueryHeapHandle RenderBackendCreateOcclusionQueryHeap(RenderBackend* backend, uint32 deviceMask, const RenderBackendOcclusionQueryHeapDesc* desc, const char* name);
	void RenderBackendDestroyOcclusionQueryHeap(RenderBackend* backend, RenderBackendOcclusionQueryHeapHandle occlusionQueryHeap);
	void RenderBackendSubmitRenderCommandLists(RenderBackend* backend, RenderCommandList** commandLists, uint32 numCommandLists);
	void RenderBackendGetRenderStatistics(RenderBackend* backend, uint32 deviceMask, RenderStatistics* statistics);
	RenderBackendRayTracingAccelerationStructureHandle RenderBackendCreateBottomLevelAS(RenderBackend* backend, uint32 deviceMask, const RenderBackendBottomLevelASDesc* desc, const char* name);
	RenderBackendRayTracingAccelerationStructureHandle RenderBackendCreateTopLevelAS(RenderBackend* backend, uint32 deviceMask, const RenderBackendTopLevelASDesc* desc, const char* name);
	RenderBackendRayTracingPipelineStateHandle RenderBackendCreateRayTracingPipelineState(RenderBackend* backend, uint32 deviceMask, const RenderBackendRayTracingPipelineStateDesc* desc, const char* name);
	RenderBackendBufferHandle RenderBackendCreateRayTracingShaderBindingTable(RenderBackend* backend, uint32 deviceMask, const RenderBackendRayTracingShaderBindingTableDesc* desc, const char* name);

	struct RenderCommandContainer
	{
		uint32 numCommands = 0;
		std::vector<RenderCommandType> types = {};
		std::vector<void*> commands = {};
	};

	class RenderCommandListBase
	{
	public:
		RenderCommandListBase(MemoryArena* arena) : arena(arena) {}
		virtual ~RenderCommandListBase() {}
		template <typename T>
		FORCEINLINE T* AllocateCommand(RenderCommandType type, uint64 size = sizeof(T))
		{
			void* data = AllocateCommandInternal(size);
			container.types.push_back(type);
			container.commands.push_back(data);
			container.numCommands++;
			return (T*)data;
		}
		FORCEINLINE RenderCommandContainer* GetCommandContainer()
		{
			return &container;
		}
	private:
		FORCEINLINE void* AllocateCommandInternal(uint64 size)
		{
			void* data = HE_ARENA_ALLOC(arena, size);
			return data;
		}
		MemoryArena* arena;
		RenderCommandContainer container;
	};

	/**
	 * Render command list encodes high level commands.
	 * The commands are stateless, which allows for fully parallel recording.
	 */
	class RenderCommandList : public RenderCommandListBase
	{
	public:
		RenderCommandList(MemoryArena* arena) : RenderCommandListBase(arena) {}
		// Copy commands
		void CopyTexture2D(RenderBackendTextureHandle srcTexture, const Offset2D& srcOffset, uint32 srcMipLevel, RenderBackendTextureHandle dstTexture, const Offset2D& dstOffset, uint32 dstMipLevel, const Extent2D extent);
		void CopyBuffer(RenderBackendBufferHandle srcBuffer, uint64 srcOffset, RenderBackendBufferHandle dstBuffer, uint64 dstOffset, uint64 bytes);
		void UpdateBuffer(RenderBackendBufferHandle buffer, uint64 offset, const void* data, uint64 size);
		// Compute commands
		void Dispatch(RenderBackendShaderHandle shader, const ShaderArguments& shaderArguments, uint32 x, uint32 y, uint32 z);
		void Dispatch2D(RenderBackendShaderHandle shader, const ShaderArguments& shaderArguments, uint32 x, uint32 y);
		/*void BuildTopLevelAS();
		void BuildBottomLevelAS();
		void UpdateTopLevelAS();
		void UpdateBottomLevelAS(); */
		void TraceRays(RenderBackendRayTracingPipelineStateHandle pipelineState, RenderBackendBufferHandle shaderBindingTable, const ShaderArguments& shaderArguments, uint32 x, uint32 y, uint32 z);
		// Graphics commands
		void SetViewports(RenderBackendViewport* viewports, uint32 numViewports);
		void SetScissors(RenderBackendScissor* scissors, uint32 numScissors);
		void Transitions(RenderBackendBarrier* transitions, uint32 numTransitions);
		void BeginRenderPass(const RenderPassInfo& renderPassInfo);
		void EndRenderPass();
		void Draw(RenderBackendShaderHandle shader, const ShaderArguments& shaderArguments, uint32 numVertices, uint32 numInstances, uint32 firstVertex, uint32 firstInstance, PrimitiveTopology topology);
		void DrawIndexed(RenderBackendShaderHandle shader, const ShaderArguments& shaderArguments, RenderBackendBufferHandle indexBuffer, uint32 numIndices, uint32 numInstances, uint32 firstIndex, int32 vertexOffset, uint32 firstInstance, PrimitiveTopology topology);
		void DrawIndirect(RenderBackendShaderHandle shader, const ShaderArguments& shaderArguments, RenderBackendBufferHandle indexBuffer, RenderBackendBufferHandle argumentBuffer, uint64 offset, uint32 numDraws, uint32 stride, PrimitiveTopology topology);
		void DrawIndexedIndirect(RenderBackendShaderHandle shader, const ShaderArguments& shaderArguments, RenderBackendBufferHandle indexBuffer, RenderBackendBufferHandle argumentBuffer, uint64 offset, uint32 numDraws, uint32 stride, PrimitiveTopology topology);
		void BeginTimingQuery(RenderBackendTimingQueryHeapHandle timingQueryHeap, uint32 region);
		void EndTimingQuery(RenderBackendTimingQueryHeapHandle timingQueryHeap, uint32 region);
		void ResetTimingQueryHeap(RenderBackendTimingQueryHeapHandle timingQueryHeap, uint32 regionStart, uint32 regionCount);
	};

	struct ShaderCompiler;
	class UIRenderer;

	struct RenderContext
	{
		MemoryArena* arena;
		RenderBackend* renderBackend;
		ShaderCompiler* shaderCompiler;
		UIRenderer* uiRenderer;
		RenderBackendTimingQueryHeapHandle timingQueryHeap;
		std::vector<RenderCommandList*> commandLists;
		// ShaderLibrary* shaderLibrary;
	};

	enum class ShaderRepresentation
	{
		DXIL,
		SPIRV,
	};

	struct ShaderCompiler
	{
		void* instance;
		bool (*CompileShader)(
			void* instance,
			std::vector<uint8> source,
			const wchar* entry,
			RenderBackendShaderStage stage,
			ShaderRepresentation representation,
			const std::vector<const wchar*>& includeDirs,
			const std::vector<const wchar*>& defines,
			ShaderBlob* outBlob);
		void (*ReleaseShaderBlob)(void* instance, ShaderBlob* blob);
	};

	void LoadShaderSourceFromFile(const char* filename, std::vector<uint8>& outData);

	bool CompileShader(
		ShaderCompiler* compiler,
		std::vector<uint8> source,
		const wchar* entry,
		RenderBackendShaderStage stage,
		ShaderRepresentation representation,
		const std::vector<const wchar*>& includeDirs,
		const std::vector<const wchar*>& defines,
		ShaderBlob* outBlob);

	void ReleaseShaderBlob(ShaderCompiler* compiler, ShaderBlob* blob);

	struct SceneView;
	class Renderer
	{
	public:
		Renderer() = default;
		virtual	~Renderer() = default;
		virtual void Render(SceneView* view) = 0;
	};
}
