export module HorizonEngine.Render.VulkanRenderBackend;

export namespace HE
{
	struct RenderBackend;
	
	enum VulkanRenderBackendCreateFlags
	{
		VULKAN_RENDER_BACKEND_CREATE_FLAGS_NONE = 0,
		VULKAN_RENDER_BACKEND_CREATE_FLAGS_VALIDATION_LAYERS = (1 << 1),
		VULKAN_RENDER_BACKEND_CREATE_FLAGS_SURFACE = (1 << 2),
	};

	typedef RenderBackend* (__stdcall* PFN_VulkanRenderBackendCreateBackend)(int flags);
	typedef void (__stdcall* PFN_VulkanRenderBackendDestroyBackend)(RenderBackend* backend);

	RenderBackend* VulkanRenderBackendCreateBackend(int flags);
	void VulkanRenderBackendDestroyBackend(RenderBackend* backend);
}
