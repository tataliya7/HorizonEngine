#pragma once

#ifndef HE_CONFIG_DEBUG
#if defined(_DEBUG) || defined(DEBUG)
#define HE_CONFIG_DEBUG
#else
#define HE_CONFIG_DEBUG
#endif
#endif

#define HE_MAKE_VERSION(major, minor, patch) ((((uint32)(major)) << 22) | (((uint32)(minor)) << 12) | ((uint32)(patch)))
#define HE_VERSION_MAJOR(version) ((uint32)(version) >> 22)
#define HE_VERSION_MINOR(version) (((uint32)(version) >> 12) & 0x3ff)
#define HE_VERSION_PATCH(version) ((uint32)(version) & 0xfff)
#define HE_VERSION_1_0_0 HE_MAKE_VERSION(1, 0, 0)

#define HE_ENGINE_NAME "Horizon Engine"
#define HE_ENGINE_VERSION HE_VERSION_1_0_0

#if defined(_MSC_VER)
#define HE_DISABLE_WARNINGS __pragma(warning(push, 0))
#define HE_ENABLE_WARNINGS __pragma(warning(pop))
#endif

#if defined(_MSC_VER)
#define HE_DLL_EXPORT __declspec(dllexport)
#else
#define HE_DLL_EXPORT __attribute__((visibility("default")))
#endif

#define HE_ENABLE_ASSERTION

#define HE_TEXT(quote) L##quote

#define HE_DEFAULT_ALIGNMENT uint64(8)

#define HE_CORE_API

#define STATIC_ASSERT(expression) static_assert(expression)

#if defined(HE_ENABLE_ASSERTION)
	#define ASSERT(expression) assert(expression)
#else
	#define ASSERT(expression)
#endif

#if !defined(NOMINMAX) && defined(HE_PLATFORM_WINDOWS)
#define NOMINMAX
#endif

#if defined(_MSC_VER)
#define FORCEINLINE __forceinline
#endif

#define ARRAY_SIZE(a) ((int)(sizeof(a) / sizeof(*(a))))

#define INVALID_ENUM_VALUE() assert(false)

#define ENUM_CLASS_OPERATORS(EnumClass) inline           EnumClass& operator|=(EnumClass& lhs, EnumClass rhs)   { return lhs = (EnumClass)((__underlying_type(EnumClass))rhs | (__underlying_type(EnumClass))rhs); } \
										inline           EnumClass& operator&=(EnumClass& lhs, EnumClass rhs)   { return lhs = (EnumClass)((__underlying_type(EnumClass))rhs & (__underlying_type(EnumClass))rhs); } \
										inline           EnumClass& operator^=(EnumClass& lhs, EnumClass rhs)   { return lhs = (EnumClass)((__underlying_type(EnumClass))rhs ^ (__underlying_type(EnumClass))rhs); } \
										inline constexpr EnumClass  operator| (EnumClass  lhs, EnumClass rhs)   { return (EnumClass)((__underlying_type(EnumClass))lhs | (__underlying_type(EnumClass))rhs); } \
										inline constexpr EnumClass  operator& (EnumClass  lhs, EnumClass rhs)   { return (EnumClass)((__underlying_type(EnumClass))lhs & (__underlying_type(EnumClass))rhs); } \
										inline constexpr EnumClass  operator^ (EnumClass  lhs, EnumClass rhs)   { return (EnumClass)((__underlying_type(EnumClass))lhs ^ (__underlying_type(EnumClass))rhs); } \
										inline constexpr bool       operator! (EnumClass  e)                    { return !(__underlying_type(EnumClass))e; } \
										inline constexpr EnumClass  operator~ (EnumClass  e)                    { return (EnumClass)~(__underlying_type(EnumClass))e; } \
										inline           bool       HAS_ANY_FLAGS(EnumClass e, EnumClass flags) { return (e & flags) != (EnumClass)0; }

#define HE_BIND_FUNCTION(func) [this](auto&&... args) -> decltype(auto) { return this->func(std::forward<decltype(args) > (args)...); }

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define ABORT() abort()

#define HE_DECLARE_HANDLE(object) typedef struct object##_T* object;

#define HE_TEXT(quote) L##quote

#ifndef HE_LOG
#define HE_LOG(level, ...)
#endif

#ifndef HE_LOG_VERBOSE
#define HE_LOG_VERBOSE(...)    ::HE::LogVerbose(__VA_ARGS__);
#endif

#ifndef HE_LOG_INFO
#define HE_LOG_INFO(...)       ::HE::LogInfo(__VA_ARGS__);
#endif

#ifndef HE_LOG_WARNING
#define HE_LOG_WARNING(...)    ::HE::LogWarning(__VA_ARGS__);
#endif

#ifndef HE_LOG_ERROR
#define HE_LOG_ERROR(...)      ::HE::LogError(__VA_ARGS__);
#endif

#ifndef HE_LOG_FATAL
#define HE_LOG_FATAL(...)      ::HE::LogFatal(__VA_ARGS__);
#endif

#define GLM_FORCE_CTOR_INIT
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#define M_PI 				  (3.1415926535897932f)	
#define M_INV_PI			  (0.3183098861837067f)
#define M_HALF_PI			  (1.5707963267948966f)
#define M_TWO_PI			  (6.2831853071795864f)
#define M_PI_SQUARED		  (9.8696044010893580f)
#define M_SQRT_PI	      	  (1.4142135623730950f)
#define SMALL_NUMBER		  (1.e-8f)
#define KINDA_SMALL_NUMBER    (1.e-4f)
#define BIG_NUMBER			  (3.4e+38f)
#define DELTA			      (0.00001f)
#define FLOAT_MAX			  (3.402823466e+38f)

#define HE_ARENA_ALLOC(arena, size)                                          (::HE::ArenaRealloc(arena, nullptr, 0,       size,    HE_DEFAULT_ALIGNMENT, __FILE__, __LINE__))
#define HE_ARENA_FREE(arena, ptr, size)                                      (::HE::ArenaRealloc(arena, ptr,     size,    0,       HE_DEFAULT_ALIGNMENT, __FILE__, __LINE__))
#define HE_ARENA_REALLOC(arena, ptr, oldSize, newSize)                       (::HE::ArenaRealloc(arena, ptr,     oldSize, newSize, HE_DEFAULT_ALIGNMENT, __FILE__, __LINE__))
#define HE_ARENA_ALIGNED_ALLOC(arena, size, alignment)                       (::HE::ArenaRealloc(arena, nullptr, 0,       size,    alignment,            __FILE__, __LINE__))
#define HE_ARENA_ALIGNED_FREE(arena, ptr, size, alignment)                   (::HE::ArenaRealloc(arena, ptr,     size,    0,       alignment,            __FILE__, __LINE__))
#define HE_ARENA_ALIGNED_REALLOC(arena, ptr, oldSize, newSize, alignment)    (::HE::ArenaRealloc(arena, ptr,     oldSize, newSize, alignment,            __FILE__, __LINE__))
