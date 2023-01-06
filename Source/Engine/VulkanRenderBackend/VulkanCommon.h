#pragma once

#include "VulkanDefinitions.h"

#ifdef HE_PLATFORM_WINDOWS
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>
