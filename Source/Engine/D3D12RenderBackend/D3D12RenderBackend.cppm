export module HorizonEngine.Render.D3D12RenderBackend;

export namespace HE
{
	struct RenderBackend;

	enum D3D12RenderBackendCreateFlags
	{
		D3D12_RENDER_BACKEND_CREATE_FLAGS_NONE = 0,
		D3D12_RENDER_BACKEND_CREATE_FLAGS_ENABLE_DEBUG_LAYER = (1 << 1),
		D3D12_RENDER_BACKEND_CREATE_FLAGS_ENABLE_GPU_BASED_VALIDATION = (1 << 2),
	};

	RenderBackend* D3D12RenderBackendCreateBackend(D3D12RenderBackendCreateFlags flags);
	void D3D12RenderBackendDestroyBackend(RenderBackend* backend);
}