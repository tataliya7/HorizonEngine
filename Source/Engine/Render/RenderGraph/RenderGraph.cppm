module;

#include <map>
#include <vector>
#include <functional>

export module HorizonEngine.Render.RenderGraph;

export import :RenderGraphHandles;
export import :RenderGraphNode;
export import :RenderGraphBlackboard;
export import "RenderGraphDefinitions.h";

import HorizonEngine.Core;
import HorizonEngine.Render.Core;

export namespace HE
{
	enum
	{
		RenderGraphPassMaxNumReadResources = 32,
		RenderGraphPassMaxNumWriteResources = 16,
		RenderGraphPassMaxCreateTransientResources = 16,
	};
	
	class RenderGraph;
	class RenderGraphPass;

	enum class RenderGraphResourceType
	{
		Texture,
		Buffer,
	};

	struct RenderGraphTextureSubresource
	{
		RenderGraphTextureSubresource() : level(0), layer(0) {}
		RenderGraphTextureSubresource(uint32 level, uint32 layer) : level(level), layer(layer) {}
		uint32 level;
		uint32 layer;
	};

	struct RenderGraphTextureSubresourceLayout
	{
		RenderGraphTextureSubresourceLayout(uint32 numMipLevels, uint32 numArrayLayers)
			: numMipLevels(numMipLevels), numArrayLayers(numArrayLayers) {}
		uint32 GetSubresourceCount() const
		{
			return numMipLevels * numArrayLayers;
		}
		inline uint32 GetSubresourceIndex(const RenderGraphTextureSubresource& subresource) const
		{
			return subresource.layer + (subresource.level * numArrayLayers);
		}
		uint32 numMipLevels;
		uint32 numArrayLayers;
	};

	struct RenderGraphTextureSubresourceRange
	{
		static const RenderGraphTextureSubresourceRange WholeRange;
		static RenderGraphTextureSubresourceRange Create(uint32 baseMipLevel, uint32 numMipLevels, uint32 baseArrayLayer, uint32 numArrayLayers)
		{
			return RenderGraphTextureSubresourceRange(baseMipLevel, numMipLevels, baseArrayLayer, numArrayLayers);
		}
		static RenderGraphTextureSubresourceRange CreateForAllSubresources()
		{
			return RenderGraphTextureSubresourceRange(0, REMAINING_MIP_LEVELS, 0, REMAINING_ARRAY_LAYERS);
		}
		static RenderGraphTextureSubresourceRange CreateForMipLevel(uint32 mipLevel)
		{
			return RenderGraphTextureSubresourceRange(mipLevel, 1, 0, REMAINING_ARRAY_LAYERS);
		}
		static RenderGraphTextureSubresourceRange CreateFromLayout(const RenderGraphTextureSubresourceLayout& layout)
		{
			return RenderGraphTextureSubresourceRange(0, layout.numMipLevels, 0, layout.numArrayLayers);
		}
		RenderGraphTextureSubresourceRange(uint32 baseMipLevel, uint32 numMipLevels, uint32 baseArrayLayer, uint32 numArrayLayers)
			: baseMipLevel(baseMipLevel), numMipLevels(numMipLevels), baseArrayLayer(baseArrayLayer), numArrayLayers(numArrayLayers) {}
		RenderGraphTextureSubresource GetMinSubresource() const
		{
			return RenderGraphTextureSubresource(baseMipLevel, baseArrayLayer);
		}
		RenderGraphTextureSubresource GetMaxSubresource() const
		{
			return RenderGraphTextureSubresource(baseMipLevel + numMipLevels, baseArrayLayer + numArrayLayers);
		}
		uint32 baseMipLevel;
		uint32 numMipLevels;
		uint32 baseArrayLayer;
		uint32 numArrayLayers;
	};

	class RenderGraphResource : public RenderGraphNode
	{
	public:
		RenderGraphResourceType GetType() const
		{
			return type;
		}
		bool IsImported() const
		{
			return imported;
		}
		bool IsExported() const
		{
			return exported;
		}
	protected:
		RenderGraphResource(const char* name, RenderGraphResourceType type)
			: RenderGraphNode(name, RenderGraphNodeType::Resource)
			, type(type) {}
		bool imported = false;
		bool exported = false;
		bool transient = false;
		bool usedByAsyncComputePass = false;
		RenderBackendResourceState initialState = RenderBackendResourceState::Undefined;
		RenderBackendResourceState finalState = RenderBackendResourceState::Undefined;
		RenderGraphPass* firstPass = nullptr;
		RenderGraphPass* lastPass = nullptr;
	private:
		friend class RenderGraph;
		friend class RenderGraphBuilder;
		const RenderGraphResourceType type;
	};

	using RenderGraphTextureDesc = RenderBackendTextureDesc;

	class RenderGraphTexture final : public RenderGraphResource
	{
	public:
		const RenderGraphTextureDesc& GetDesc() const
		{
			return desc;
		}
		RenderBackendTextureHandle GetRenderBackendTexture() const
		{
			return texture;
		}
		RenderGraphTextureSubresourceLayout GetSubresourceLayout() const
		{
			return subresourceLayout;
		}
		RenderGraphTextureSubresourceRange GetSubresourceRange() const
		{
			return RenderGraphTextureSubresourceRange::CreateFromLayout(subresourceLayout);
		}
		bool HasRenderBackendTexture() const
		{
			return texture != RenderBackendTextureHandle::NullHandle;
		}
	private:
		friend class RenderGraph;
		RenderGraphTexture(const char* name, const RenderGraphTextureDesc& desc)
			: RenderGraphResource(name, RenderGraphResourceType::Texture)
			, desc(desc)
			, subresourceLayout(desc.mipLevels, desc.arrayLayers)
		{

		}
		void SetRenderBackendTexture(RenderBackendTextureHandle texture, RenderBackendResourceState initialState)
		{
			this->texture = texture;
			this->initialState = initialState;
			this->tempState = initialState;
			this->finalState = initialState;
		}
		const RenderGraphTextureDesc desc;
		RenderBackendResourceState tempState = RenderBackendResourceState::Undefined;
		RenderGraphTextureSubresourceLayout subresourceLayout;
		RenderBackendTextureHandle texture = RenderBackendTextureHandle::NullHandle;
	};

	enum class RenderGraphResourceViewType
	{
		TextureSRV,
		TextureUAV,
		BufferSRV,
		BufferUAV,
	};

	class RenderGraphResourceView
	{
	public:
		RenderGraphResourceViewType GetType() const
		{
			return type;
		}
	protected:
		RenderGraphResourceView(const char* name, RenderGraphResourceViewType type)
			: name(name), type(type) {}
	private:
		const char* name;
		const RenderGraphResourceViewType type;
	};

	class RenderGraphShaderResourceView : public RenderGraphResourceView
	{
	public:
		//uint32 GetDescripotrIndex() const { return static_cast<FRHIShaderResourceView*>(FRDGResource::GetRHI()); }
	protected:
		RenderGraphShaderResourceView(const char* name, RenderGraphResourceViewType type) : RenderGraphResourceView(name, type) {}
	};

	class RenderGraphUnorderedAccessView : public RenderGraphResourceView
	{
	public:
		//uint32 GetDescripotrIndex() const { return static_cast<FRHIShaderResourceView*>(FRDGResource::GetRHI()); }
	protected:
		RenderGraphUnorderedAccessView(const char* name, RenderGraphResourceViewType type) : RenderGraphResourceView(name, type) {}
	};

	struct RenderGraphTextureSRVDesc
	{
		static RenderGraphTextureSRVDesc Create(RenderGraphTextureHandle texture, uint32 baseMipLevel, uint32 mipLevelCount, uint32 baseArrayLayer, uint32 arrayLayerCount)
		{
			return RenderGraphTextureSRVDesc(texture, baseMipLevel, mipLevelCount, baseArrayLayer, arrayLayerCount);
		}
		static RenderGraphTextureSRVDesc CreateForMipLevel(RenderGraphTextureHandle texture, uint32 mipLevel, uint32 baseArrayLayer = 0, uint32 arrayLayerCount = 1)
		{
			return RenderGraphTextureSRVDesc(texture, mipLevel, 1, baseArrayLayer, arrayLayerCount);
		}
		RenderGraphTextureSRVDesc(RenderGraphTextureHandle texture, uint32 baseMipLevel, uint32 numMipLevels, uint32 baseArrayLayer, uint32 numArrayLayers)
			: texture(texture)
			, baseMipLevel(baseMipLevel)
			, numMipLevels(numMipLevels)
			, baseArrayLayer(baseArrayLayer)
			, numArrayLayers(numArrayLayers) {}
		RenderGraphTextureHandle texture;
		uint32 baseMipLevel;
		uint32 numMipLevels;
		uint32 baseArrayLayer;
		uint32 numArrayLayers;
	};

	class RenderGraphTextureSRV final : public RenderGraphShaderResourceView
	{
	public:
		const RenderGraphTextureSRVDesc& GetDesc() const
		{
			return desc;
		}
		RenderGraphTextureSubresourceRange GetSubresourceRange() const
		{
			return RenderGraphTextureSubresourceRange::Create(desc.baseMipLevel, desc.numMipLevels, desc.baseArrayLayer, desc.numArrayLayers);
		}
	private:
		friend class RenderGraph;
		RenderGraphTextureSRV(const char* name, const RenderGraphTextureSRVDesc& desc)
			: RenderGraphShaderResourceView(name, RenderGraphResourceViewType::TextureSRV), desc(desc) {}
		const RenderGraphTextureSRVDesc desc;
	};

	struct RenderGraphTextureUAVDesc
	{
		static RenderGraphTextureUAVDesc Create(RenderGraphTextureHandle texture, uint32 mipLevel = 0)
		{
			return RenderGraphTextureUAVDesc(texture, mipLevel);
		}
		RenderGraphTextureUAVDesc() = default;
		RenderGraphTextureUAVDesc(RenderGraphTextureHandle texture, uint32 mipLevel)
			: texture(texture), mipLevel(mipLevel) {}
		RenderGraphTextureHandle texture = RenderGraphTextureHandle::NullHandle;
		uint32 mipLevel = 0;
	};

	class RenderGraphTextureUAV final : public RenderGraphUnorderedAccessView
	{
	public:
		const RenderGraphTextureUAVDesc& GetDesc() const
		{
			return desc;
		}
		RenderGraphTextureSubresourceRange GetSubresourceRange() const
		{
			return RenderGraphTextureSubresourceRange::CreateForMipLevel(desc.mipLevel);
		}
	private:
		friend class RenderGraph;
		RenderGraphTextureUAV(const char* name, const RenderGraphTextureUAVDesc& desc)
			: RenderGraphUnorderedAccessView(name, RenderGraphResourceViewType::TextureUAV), desc(desc) {}
		const RenderGraphTextureUAVDesc desc;
	};

	using RenderGraphBufferDesc = RenderBackendBufferDesc;

	class RenderGraphBuffer final : public RenderGraphResource
	{
	public:
		const RenderGraphBufferDesc& GetDesc() const
		{
			return desc;
		}
		RenderBackendBufferHandle GetRenderBackendBuffer() const
		{
			return buffer;
		}
	private:
		friend class RenderGraph;
		RenderGraphBuffer(const char* name, const RenderGraphBufferDesc& desc)
			: RenderGraphResource(name, RenderGraphResourceType::Buffer), desc(desc) {}
		void SetRenderBackendBuffer(RenderBackendBufferHandle buffer, RenderBackendResourceState initialState)
		{
			this->imported = true;
			this->buffer = buffer;
			this->initialState = initialState;
			this->finalState = initialState;
		}
		const RenderGraphBufferDesc desc;
		RenderBackendBufferHandle buffer;
	};

	struct RenderGraphBufferUAVDesc
	{
		static RenderGraphBufferUAVDesc Create(RenderGraphBufferHandle buffer)
		{
			return RenderGraphBufferUAVDesc(buffer);
		}
		RenderGraphBufferUAVDesc() = default;
		RenderGraphBufferUAVDesc(RenderGraphBufferHandle buffer) : buffer(buffer) {}
		RenderGraphBufferHandle buffer = RenderGraphBufferHandle::NullHandle;
	};

	class RenderGraphBufferUAV final : public RenderGraphUnorderedAccessView
	{
	public:
		const RenderGraphBufferUAVDesc& GetDesc() const
		{
			return desc;
		}
	private:
		friend class RenderGraph;
		RenderGraphBufferUAV(const char* name, const RenderGraphBufferUAVDesc& desc)
			: RenderGraphUnorderedAccessView(name, RenderGraphResourceViewType::BufferUAV), desc(desc) {}
		const RenderGraphBufferUAVDesc desc;
	};

	struct RenderGraphPersistentTexture
	{
		bool active;
		RenderBackendTextureHandle texture;
		RenderBackendTextureDesc desc;
		RenderBackendResourceState initialState;
	};

	struct RenderGraphPersistentBuffer
	{
		bool active;
		RenderBackendBufferHandle buffer;
		RenderBackendBufferDesc desc;
		RenderBackendResourceState initialState;
	};

	class RenderGraphResourcePool
	{
	public:
		void Tick();
		void CacheTexture(const RenderGraphPersistentTexture& texture);
		void ReleaseTexture(RenderBackendTextureHandle texture);
		RenderBackendTextureHandle FindOrCreateTexture(RenderBackend* backend, const RenderBackendTextureDesc* desc, const char* name);
	private:
		friend class RenderGraph;
		std::vector<RenderGraphPersistentTexture> allocatedTextures;
		uint32 frameCounter = 0;
	};

	extern RenderGraphResourcePool* GRenderGraphResourcePool;

	class RenderGraphBuilder
	{
	public:
		RenderGraphBuilder(RenderGraph* renderGraph, RenderGraphPass* pass)
			: renderGraph(renderGraph), pass(pass) {}
		~RenderGraphBuilder() = default;
		RenderGraphTextureHandle CreateTransientTexture(const RenderGraphTextureDesc& desc, const char* name);
		RenderGraphBufferHandle CreateTransientBuffer(const RenderGraphBufferDesc& desc, const char* name);
		RenderGraphTextureHandle ReadTexture(RenderGraphTextureHandle handle, RenderBackendResourceState initalState, const RenderGraphTextureSubresourceRange& range = RenderGraphTextureSubresourceRange::WholeRange);
		RenderGraphTextureHandle WriteTexture(RenderGraphTextureHandle handle, RenderBackendResourceState initalState, const RenderGraphTextureSubresourceRange& range = RenderGraphTextureSubresourceRange::WholeRange);
		RenderGraphTextureHandle ReadWriteTexture(RenderGraphTextureHandle handle, RenderBackendResourceState initalState, const RenderGraphTextureSubresourceRange& range = RenderGraphTextureSubresourceRange::WholeRange);
		RenderGraphBufferHandle ReadBuffer(RenderGraphBufferHandle handle, RenderBackendResourceState initalState);
		RenderGraphBufferHandle WriteBuffer(RenderGraphBufferHandle handle, RenderBackendResourceState initalState);
		RenderGraphBufferHandle ReadWriteBuffer(RenderGraphBufferHandle handle, RenderBackendResourceState initalState);
		void BindColorTarget(
			uint32 slot,
			RenderGraphTextureHandle handle,
			RenderTargetLoadOp loadOp,
			RenderTargetStoreOp storeOp, 
			uint32 mipLevel = 0,
			uint32 arraylayer = 0);
		void BindDepthTarget(
			RenderGraphTextureHandle handle,
			RenderTargetLoadOp depthLoadOp,
			RenderTargetStoreOp depthStoreOp,
			uint32 mipLevel = 0,
			uint32 arraylayer = 0);
		void BindDepthStencilTarget(
			RenderGraphTextureHandle handle, 
			RenderTargetLoadOp depthLoadOp, 
			RenderTargetStoreOp depthStoreOp, 
			RenderTargetLoadOp stencilLoadOp, 
			RenderTargetStoreOp stencilStoreOp,
			uint32 mipLevel = 0,
			uint32 arraylayer = 0);
	private:
		RenderGraph* const renderGraph;
		RenderGraphPass* const pass;
	};

	class RenderGraphRegistry
	{
	public:
		RenderGraphRegistry(RenderGraph* renderGraph, RenderGraphPass* pass)
			: renderGraph(renderGraph), pass(pass) {}
		RenderGraphTexture* GetTexture(RenderGraphTextureHandle handle) const;
		RenderGraphBuffer* GetBuffer(RenderGraphBufferHandle handle) const;
		RenderGraphTexture* GetImportedTexture(RenderBackendTextureHandle handle) const;
		RenderGraphBuffer* GetImportedBuffer(RenderBackendBufferHandle handle) const;
		const RenderBackendTextureDesc& GetTextureDesc(RenderGraphTextureHandle handle) const;
		const RenderBackendBufferDesc& GetBufferDesc(RenderGraphBufferHandle handle) const;
		RenderBackendTextureHandle GetRenderBackendTexture(RenderGraphTextureHandle handle) const;
		RenderBackendBufferHandle GetRenderBackendBuffer(RenderGraphBufferHandle handle) const;
	private:
		RenderGraph* const renderGraph;
		RenderGraphPass* const pass;
	};

	enum class RenderGraphPassFlags : uint8
	{
		None = 0,
		Raster = (1 << 0),
		Compute = (1 << 1),
		AsyncCompute = (1 << 2),
		RayTrace = (1 << 3),
		NeverGetCulled = (1 << 4),
		SkipRenderPass = (1 << 6),
	};
	ENUM_CLASS_OPERATORS(RenderGraphPassFlags);

	class RenderGraphPass : public RenderGraphNode
	{
	public:
		virtual ~RenderGraphPass() = default;
		bool IsAsyncCompute() const
		{
			return HAS_ANY_FLAGS(flags, RenderGraphPassFlags::AsyncCompute);
		}
		RenderGraphPassFlags GetFlags() const
		{
			return flags;
		}
		void Graphviz(std::stringstream& stream) const;
		virtual void Execute(RenderGraphRegistry& registry, RenderCommandList& commandList) = 0;
	protected:
		friend class RenderGraph;
		friend class RenderGraphBuilder;

		RenderGraphPass(const char* name, RenderGraphPassFlags flags)
			: RenderGraphNode(name, RenderGraphNodeType::Pass)
			, flags(flags) {}

		RenderGraphPassFlags flags;

#if HE_MGPU
		RenderBackendGpuMask gpuMask;
#endif

		struct TextureState
		{
			RenderGraphTexture* texture;
			RenderBackendResourceState state;
			RenderGraphTextureSubresourceRange subresourceRange;
		};

		struct BufferState
		{
			RenderGraphBuffer* buffer;
			RenderBackendResourceState state;
		};

		std::vector<TextureState> textureStates;
		std::vector<BufferState> bufferStates;

		std::vector<RenderBackendBarrier> barriers;

		struct ColorRenderTarget
		{
			RenderGraphTextureHandle texture;
			uint32 mipLevel;
			uint32 arrayLayer;
			RenderTargetLoadOp loadOp;
			RenderTargetStoreOp storeOp;
		};
		struct DepthStencilRenderTarget
		{
			RenderGraphTextureHandle texture;
			uint32 mipLevel;
			uint32 arrayLayer;
			RenderTargetLoadOp depthLoadOp;
			RenderTargetStoreOp depthStoreOp;
			RenderTargetLoadOp stencilLoadOp;
			RenderTargetStoreOp stencilStoreOp;
		};
		ColorRenderTarget colorTargets[MaxNumSimultaneousColorRenderTargets];
		DepthStencilRenderTarget depthStentcilTarget;
	};

	class RenderGraphLambdaPass : public RenderGraphPass
	{
	public:
		using Lambda = std::function<void(RenderGraphRegistry&, RenderCommandList&)>;
		~RenderGraphLambdaPass() = default;
	private:
		friend class RenderGraph;
		RenderGraphLambdaPass(const char* name, RenderGraphPassFlags flags) : RenderGraphPass(name, flags) {}
		void SetExecuteCallback(Lambda&& execute) { executeCallback = std::move(execute); }
		void Execute(RenderGraphRegistry& registry, RenderCommandList& commandList) override
		{
			ASSERT(executeCallback);
			executeCallback(registry, commandList);
		}
		Lambda executeCallback;
	};

	class RenderGraph
	{
	public:

		RenderGraph(MemoryArena* arena);
		RenderGraph(const RenderGraph& other) = delete;
		virtual ~RenderGraph();

		template<typename SetupLambdaType>
		void AddPass(const char* name, RenderGraphPassFlags flags, SetupLambdaType setup);

		void Execute(RenderContext* contex);

		void Clear();

		/**
		 * @brief Create a string using the Graphviz format.
		 * @note Compile() should be called before calling this function.
		 * @return std::string in the Graphviz format.
		 */
		std::string Graphviz() const;

		RenderGraphTextureHandle CreateTexture(const RenderGraphTextureDesc& desc, const char* name);
		RenderGraphBufferHandle CreateBuffer(const RenderGraphBufferDesc& desc, const char* name);
		//RenderGraphTextureSRVHandle CreateTextureSRV(RenderGraphTextureHandle texture, const RenderGraphTextureSRVDesc& desc);
		//RenderGraphTextureUAVHandle CreateTextureUAV(RenderGraphTextureHandle texture, uint32 mipLevel);
		RenderGraphTextureHandle ImportExternalTexture(RenderBackendTextureHandle renderBackendTexture, const RenderBackendTextureDesc& desc, RenderBackendResourceState initialState, char const* name);
		RenderGraphBufferHandle ImportExternalBuffer(RenderBackendBufferHandle renderBackendBuffer, const RenderBackendBufferDesc& desc, RenderBackendResourceState initialState, char const* name);
		void ExportTextureDeferred(RenderGraphTextureHandle handle, RenderGraphPersistentTexture* persistentTexture);

		RenderGraphBlackboard blackboard;   

	private:

		friend class RenderGraphBuilder;
		friend class RenderGraphRegistry;
	
		bool Compile();

		void* Alloc(uint32 size)
		{
			return HE_ARENA_ALLOC(arena, size);
		}

		void* AlignedAlloc(uint32 size, uint32 alignment)
		{
			return HE_ARENA_ALIGNED_ALLOC(arena, size, alignment);
		}

		template <typename ObjectType, typename... Args>
		FORCEINLINE ObjectType* AllocObject(Args&&... args)
		{
			ObjectType* result = (ObjectType*)HE_ARENA_ALLOC(arena, sizeof(ObjectType));
			ASSERT(result);
			result = new(result) ObjectType(std::forward<Args>(args)...);
			return result;
		}

		MemoryArena* arena;

		RenderGraphDAG dag;

		std::vector<RenderGraphPass*> passes;

		std::vector<RenderGraphTexture*> textures;
		std::vector<RenderGraphBuffer*> buffers;

		std::map<RenderBackendTextureHandle, RenderGraphTextureHandle> importedTextures;
		std::map<RenderBackendBufferHandle, RenderGraphBufferHandle> importedBuffers;

		struct RenderGraphExportedTexture
		{
			RenderGraphTexture* texture;
			RenderGraphPersistentTexture* persistentTexture;
		};

		std::vector<RenderGraphExportedTexture> exportedTextures;

		struct RenderGraphExportedBuffer
		{

		};

		std::vector<RenderGraphExportedBuffer> exportedBuffers;
	};

	template<typename SetupLambdaType>
	void RenderGraph::AddPass(const char* name, RenderGraphPassFlags flags, SetupLambdaType setup)
	{
		RenderGraphLambdaPass* pass = AllocObject<RenderGraphLambdaPass>(name, flags);
		RenderGraphBuilder builder(this, pass);
		const auto& execute = setup(builder);
		pass->SetExecuteCallback(std::move(execute));
		passes.emplace_back(pass);
		dag.RegisterNode(pass);
	}
}
