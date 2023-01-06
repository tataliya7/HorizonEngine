export module HorizonEngine.Render.RenderGraph:RenderGraphHandles;

import HorizonEngine.Core;

export namespace HE
{
	class RenderGraphHandleBase
	{
	public:
		RenderGraphHandleBase() : index(InvalidIndex), version(0) {}
		explicit RenderGraphHandleBase(uint64 value) : index((uint32)(value >> 32)), version((uint32)value) {}
		explicit RenderGraphHandleBase(uint32 index, uint32 version = 0) : index(index), version(version) {}
		FORCEINLINE bool IsNullHandle()  const { return index == InvalidIndex; }
		FORCEINLINE operator bool() const { return !IsNullHandle(); }
		FORCEINLINE bool operator==(const RenderGraphHandleBase& rhs) const { return ((index == rhs.index) && (version == rhs.version)); }
		FORCEINLINE bool operator!=(const RenderGraphHandleBase& rhs) const { return ((index != rhs.index) || (version != rhs.version)); }
		FORCEINLINE RenderGraphHandleBase& operator=(const RenderGraphHandleBase& rhs) = default;
		FORCEINLINE RenderGraphHandleBase& operator++() { index++; return *this; }
		FORCEINLINE RenderGraphHandleBase& operator--() { index--; return *this; }
		FORCEINLINE uint32 GetIndex() const { return index; }
		FORCEINLINE uint32 GetVersion() const { return version; }
	private:
		static const uint32 InvalidIndex = UINT32_MAX;
		uint32 index;
		uint32 version;
	};

	template<typename ObjectType>
	class RenderGraphHandle : public RenderGraphHandleBase
	{
	public:
		static const RenderGraphHandle NullHandle;
		static RenderGraphHandle CreateNewVersion(RenderGraphHandle oldHandle)
		{
			return RenderGraphHandle(oldHandle.GetIndex(), oldHandle.GetVersion() + 1);
		}
		RenderGraphHandle() = default;
		explicit RenderGraphHandle(uint64 value) : RenderGraphHandleBase(value) {}
		explicit RenderGraphHandle(uint32 index, uint32 version = 0) : RenderGraphHandleBase(index, version) {}
	};

	template<typename ObjectType>
	const RenderGraphHandle<ObjectType> RenderGraphHandle<ObjectType>::NullHandle = RenderGraphHandle<ObjectType>();

	class RenderGraphTexture;
	using RenderGraphTextureHandle = RenderGraphHandle<RenderGraphTexture>;

	class RenderGraphBuffer;
	using RenderGraphBufferHandle = RenderGraphHandle<RenderGraphBuffer>;

	class RenderGraphTextureSRV;
	using RenderGraphTextureSRVHandle = RenderGraphHandle<RenderGraphTextureSRV>;

	class RenderGraphTextureUAV;
	using RenderGraphTextureUAVHandle = RenderGraphHandle<RenderGraphTextureUAV>;

	class RenderGraphBufferUAV;
	using RenderGraphBufferUAVHandle = RenderGraphHandle<RenderGraphBufferUAV>;
}