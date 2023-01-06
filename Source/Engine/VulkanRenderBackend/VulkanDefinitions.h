#pragma once

/// Enable / disable the validations layers by default.
#ifndef VULKAN_DEFAULT_ENABLE_VALIDATION_LAYERS
#if defined(_DEBUG)|| defined(DEBUG) || defined(HE_DEBUG_MODE)
#define VULKAN_DEFAULT_ENABLE_VALIDATION_LAYERS true
#else
#define VULKAN_DEFAULT_ENABLE_VALIDATION_LAYERS false
#endif
#endif

#ifndef VULKAN_ALLOCATION_CALLBACKS
#define VULKAN_ALLOCATION_CALLBACKS nullptr
#endif

#define VULKAN_API_VERSION VK_API_VERSION_1_3
#define VULKAN_API_MAJOR_VERSION 1
#define VULKAN_API_MINOR_VERSION 3
#define VULKAN_API_PATCH_VERSION 204
