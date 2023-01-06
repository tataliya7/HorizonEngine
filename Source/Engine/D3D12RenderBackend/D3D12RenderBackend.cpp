module;

#include "D3D12Common.h"
#include "D3D12MemAlloc.h"

//#pragma comment(lib, "d3d12.lib")
//#pragma comment(lib, "dxgi.lib")
//#pragma comment(lib, "dxguid.lib")

module HorizonEngine.Render.D3D12RenderBackend;

import HorizonEngine.Core;

#define D3D12_CHECK(func)                                                      \
	do                                                                         \
	{                                                                          \
		HRESULT hr = (func);                                                   \
		if (!SUCCEEDED(hr))                                                    \
		{                                                                      \
			HE_LOG_ERROR("{}: Failed with HRESULT: {}", #func, (uint32)hr);    \
			ASSERT(false);                                                     \
		}                                                                      \
	} while (0)

#define D3D12_SAFE_RELEASE(ptr)   if(ptr) { (ptr)->Release(); (ptr) = NULL; }

namespace HE
{
// namespace D3D12RenderBackend
//{
	struct D3D12Texture
	{

	};

	struct D3D12Buffer
	{

	};

	struct D3D12Sampler
	{

	};

    struct D3D12SwapChain
	{
		IDXGISwapChain3* swapChain;
		std::vector<ID3D12Resource> buffers;
	};
	
	struct D3D12CommandQueue
	{
		ID3D12CommandQueue* queue;
		uint32 familyIndex;
		uint32 queueIndex;
	};

	struct D3D12CommandList
	{

	};

	struct D3D12Aaptor
	{
		IDXGIAdapter4*                    dxgiAdaptor;
		D3D_FEATURE_LEVEL                 maxSupportedFeatureLevel;
		D3D12_FEATURE_DATA_D3D12_OPTIONS  featureDataOptions;
		D3D12_FEATURE_DATA_D3D12_OPTIONS1 featureDataOptions1;
		SIZE_T                            dedicatedVideoMemory;
		char                              vendorId[];
		char                              deviceId[];
		char                              revisionId[];
		char                              name[];
	};

	struct D3D12Device
	{
		ID3D12Device5* d3d12Device;
	};

	/*using PFN_CREATE_DXGI_FACTORY_2 = decltype(&CreateDXGIFactory2);
	using PFN_DXGI_GET_DEBUG_INTERFACE_1 = decltype(&DXGIGetDebugInterface1);

	static PFN_CREATE_DXGI_FACTORY_2 CreateDXGIFactory2 = nullptr;
	static PFN_DXGI_GET_DEBUG_INTERFACE_1 DXGIGetDebugInterface1 = nullptr;
	static PFN_D3D12_CREATE_DEVICE D3D12CreateDevice = nullptr;
	static PFN_D3D12_CREATE_VERSIONED_ROOT_SIGNATURE_DESERIALIZER D3D12CreateVersionedRootSignatureDeserializer = nullptr;*/

	struct D3D12RenderBackend
	{
		IDXGIFactory6* dxgiFactory;
		ID3D12Debug* d3d12Debug;
		IDXGIInfoQueue* dxgiInfoQueue;
		D3D12Aaptor dxgiAdapters[MaxNumRenderDevices];
		D3D12Device devices[MaxNumRenderDevices];
	};
	
	static void CreateRenderDevices(void* instance, PhysicalDeviceID* physicalDeviceIDs, uint32 numPhysicalDevices, uint32* outDeviceMasks)
	{
		D3D12RenderBackend* backend = (D3D12RenderBackend*)instance;

		for (uint32 i = 0; i < numPhysicalDevices; i++)
		{
			D3D12Device& device = backend->devices[i];
			D3D12Aaptor& adptor = backend->dxgiAdapters[0];

			D3D12_CHECK(D3D12CreateDevice(adptor.dxgiAdaptor.Get(), adptor.maxSupportedFeatureLevel, IID_PPV_ARGS(device.d3d12Device.GetAddressOf())));

			if (device.Init(backend, physicalDevice, bindlessConfig))
			{
				outDeviceMasks[i] = device.GetDeviceMask();
			}
			else
			{
				outDeviceMasks[i] = 0;
			}
		}

		for ()
		{
			dxgiAdapter = backend;
			featurelevel = D3D_FEATURE_LEVEL_12_1;
			backend->D3D12CreateDevice(dxgiAdapter, featurelevel,  IID_PPV_ARGS(&device));
			
			// Create Graphics Command Queue
			{
				D3D12_COMMAND_QUEUE_DESC queueDesc = {
					.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
					.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
					.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
					.NodeMask = 0;	
				};
				D3D12_CHECK(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queues[QUEUE_FAMILY_INDEX_GRAPHICS].queue)));
			}
			// Create Compute Command Queue
			{
				queues[QUEUE_FAMILY_INDEX_COMPUTE].desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
				queues[QUEUE_FAMILY_INDEX_COMPUTE].desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
				queues[QUEUE_FAMILY_INDEX_COMPUTE].desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
				queues[QUEUE_FAMILY_INDEX_COMPUTE].desc.NodeMask = 0;
				D3D12_CHECK(device->CreateCommandQueue(&queues[QUEUE_COMPUTE].desc, IID_PPV_ARGS(&queues[QUEUE_FAMILY_INDEX_COMPUTE].queue)));
			}
		}
	}

	static void DestroyRenderDevices()
	{

	}

	static void CreateSwapChain()
	{

	}

	static void DestroySwapChain()
	{

	}

	static RenderBackendTextureHandle CreateTexture(void* instance, uint32 deviceMask, const RenderBackendTextureDesc* desc, const char* name)
	{
		D3D12RenderBackend* backend = (D3D12RenderBackend*)instance;
		// for ()
		{
			auto& device = device[0];
			D3D12Texture texture = {
				.width = desc->width,
				.height = desc->height,
			};
			D3D12_RESOURCE_DESC desc = {
				.Format = _ConvertFormat(desc->format);
				.Width = desc->width;
				.Height = desc->height;
				.MipLevels = desc->mipLevels;
				.DepthOrArraySize = (UINT16)desc->array_size;
				.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
				.SampleDesc.Count = desc->samples;
				.SampleDesc.Quality = 0;
				.Alignment = 0;
				.Flags = D3D12_RESOURCE_FLAG_NONE;
			};

			D3D12_CHECK(allocationhandler->allocator->CreateResource(
				&allocationDesc,
				&resourcedesc,
				resourceState,
				useClearValue ? &optimizedClearValue : nullptr,
				&internal_state->allocation,
				IID_PPV_ARGS(&internal_state->resource)
			));
		}
		return handle;
	}

	static void DestroyTexture()
	{

	}

	static RenderBackendBufferHandle CreateBuffer(void* instance, uint32 deviceMask, const RenderBackendBufferDesc* desc, const char* name)
	{
		D3D12RenderBackend* backend = (D3D12RenderBackend*)instance;
		D3D12Device& device = backend->devices[0];
		// for ()
		{
			D3D12MA::ALLOCATION_DESC allocationDesc = {};
			allocationDesc.HeapType = D3D12_HEAP_TYPE_DEFAULT;
			if (desc->usage == Usage::READBACK)
			{
				allocationDesc.HeapType = D3D12_HEAP_TYPE_READBACK;
				resourceState = D3D12_RESOURCE_STATE_COPY_DEST;
				resourceDesc.Flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
			}
			else if (desc->usage == Usage::UPLOAD)
			{
				allocationDesc.HeapType = D3D12_HEAP_TYPE_UPLOAD;
				resourceState = D3D12_RESOURCE_STATE_GENERIC_READ;
			}

			Flags = D3D12_RESOURCE_FLAG_NONE;

			D3D12_RESOURCE_DESC resourceDesc = {
				.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
				.Format = DXGI_FORMAT_UNKNOWN,
				.Width = alignedSize,
				.Height = 1,
				.MipLevels = 1,
				.SampleDesc = { .Count = 1, .Quality = 0 },
				.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
				.DepthOrArraySize = 1,
				.Alignment = 0,
				.Flags = D3D12_RESOURCE_FLAG_NONE,
			}

			D3D12_RESOURCE_STATES resourceState = D3D12_RESOURCE_STATE_COMMON;
			D3D12_CHECK(allocationhandler->allocator->CreateResource(
				&allocationDesc,
				&resourceDesc,
				resourceState,
				nullptr,
				&internal_state->allocation,
				IID_PPV_ARGS(&internal_state->resource)
			);
			internal_state->gpuAddress = internal_state->resource->GetGPUVirtualAddress();
		}
		return handle;
	}
	
	static void DestroyBuffer()
	{

	}

	/*static RenderBackendSamplerHandle CreateSampler(void* instance, uint32 deviceMask, const RenderBackendSamplerDesc* desc, const char* name)
	{
		D3D12RenderBackend* backend = (D3D12RenderBackend*)instance;
		handle = handleManager.AllocateRenderHandle<RenderBackendSamplerHandle>();
		for ()
		{
			D3D12_SAMPLER_DESC samplerdesc = {
				.Filter = ToD3D12Filter(desc->filter);
				.AddressU = _ConvertTextureAddressMode(desc->address_u);
				.AddressV = _ConvertTextureAddressMode(desc->address_v);
				.AddressW = _ConvertTextureAddressMode(desc->address_w);
				.MipLODBias = desc->mip_lod_bias;
				.MaxAnisotropy = desc->max_anisotropy;
				.ComparisonFunc = _ConvertComparisonFunc(desc->comparison_func);
			};
		}
		return handle;
	}

	static void DestroySampler()
	{

	}

	static void CreateTextureView()
	{

	}

	static void GetOrCreateGraphicsPipelineState()
	{

	}

	static void GetOrCreateComputePipelineState()
	{

	}*/
//}
}

namespace HE
{
	RenderBackend* D3D12RenderBackendCreateBackend(D3D12RenderBackendCreateFlags flags)
	{
		/*HMODULE dxgiDLL = LoadLibraryEx(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
		if (dxgiDLL == nullptr)
		{
			HE_LOG_ERROR("");
			return nullptr;
		}

		HMODULE d3d12DLL = LoadLibraryEx(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
		if (d3d12DLL == nullptr)
		{
			HE_LOG_ERROR("");
			return nullptr;
		}

		CreateDXGIFactory2 = (PFN_CREATE_DXGI_FACTORY_2)GetProcAddress(dxgiDLL, "CreateDXGIFactory2");
		DXGIGetDebugInterface1 = (PFN_DXGI_GET_DEBUG_INTERFACE_1)GetProcAddress(dxgiDLL, "DXGIGetDebugInterface1");
		D3D12CreateDevice = (PFN_D3D12_CREATE_DEVICE)GetProcAddress(d3d12DLL, "D3D12CreateDevice");
		D3D12CreateVersionedRootSignatureDeserializer = (PFN_D3D12_CREATE_VERSIONED_ROOT_SIGNATURE_DESERIALIZER)GetProcAddress(d3d12DLL, "D3D12CreateVersionedRootSignatureDeserializer");*/

		D3D12RenderBackend* renderBackend = new D3D12RenderBackend();
		bool enableDebugLayer = flags & D3D12_RENDER_BACKEND_CREATE_FLAGS_ENABLE_DEBUG_LAYER;
		if (enableDebugLayer)
		{
			//auto D3D12GetDebugInterface = (PFN_D3D12_GET_DEBUG_INTERFACE)GetProcAddress(d3d12DLL, "D3D12GetDebugInterface");
			//if (D3D12GetDebugInterface)
			{
				auto& d3d12Debug = renderBackend->d3d12Debug;
				if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&d3d12Debug))))
				{
					d3d12Debug->EnableDebugLayer();
					if (flags & D3D12_RENDER_BACKEND_CREATE_FLAGS_ENABLE_GPU_BASED_VALIDATION)
					{
						ComPtr<ID3D12Debug1> d3dDebug1;
						if (SUCCEEDED(d3d12Debug.As(&d3dDebug1)))
						{
							d3dDebug1->SetEnableGPUBasedValidation(TRUE);
						}
					}
				}
			}
			/*else
			{
				HE_LOG_ERROR("");
				return nullptr;
			}
			*/
			auto& dxgiInfoQueue = renderBackend->dxgiInfoQueue;
			if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&dxgiInfoQueue))))
			{
				dxgiInfoQueue->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION, true);
				dxgiInfoQueue->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR, true);
				dxgiInfoQueue->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_WARNING, true);

				DXGI_INFO_QUEUE_MESSAGE_ID hide[] =
				{
					80 /* IDXGISwapChain::GetContainingOutput: The swapchain's adapter does not control the output on which the swapchain's window resides. */,
				};
				DXGI_INFO_QUEUE_FILTER filter = {};
				filter.DenyList.NumIDs = static_cast<UINT>(std::size(hide));
				filter.DenyList.pIDList = hide;
				dxgiInfoQueue->AddStorageFilterEntries(DXGI_DEBUG_DXGI, &filter);
			}
		}

		UINT flags = enableDebugLayer ? DXGI_CREATE_FACTORY_DEBUG : 0;
		D3D12_CHECK(CreateDXGIFactory2(flags, IID_PPV_ARGS(&renderBackend->dxgiFactory)));

		return;
	}

	void D3D12RenderBackendDestroyBackend(RenderBackend* backend)
	{
		RenderBackendDestroyRenderDevices(backend);
		D3D12RenderBackend* d3d12RenderBackend = (D3D12RenderBackend*)backend->instance;
		D3D12_SAFE_RELEASE(d3d12RenderBackend->dxgiFactory);
		delete d3d12RenderBackend;
		delete backend;
	}
}