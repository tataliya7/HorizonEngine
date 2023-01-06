#pragma once

#include "StlHeaders.h"
#include <memory>

namespace Ecila
{
#ifdef DOUBLE_PRECISION
	using Real = double;
	#define REAL_INFINITY std::numeric_limits<double>::infinity()
#else
	using Real = float;
	#define REAL_INFINITY std::numeric_limits<float>::infinity()
#endif

	using int8 = signed char;
	using uint8 = unsigned char;
	using int16 = short;
	using uint16 = unsigned short;
	using int32 = int;
	using uint32 = unsigned int;
	using int64 = long long;
	using uint64 = unsigned long long;

	using String = std::string;
	using WString = std::wstring;

	template<class T>
	using UniquePtr = std::unique_ptr<T>;

	template<class T>
	using SharedPtr = std::shared_ptr<T>;

	template<class T>
	using SharedConstPtr = std::shared_ptr<const T>;

	class CameraBase;
	class Framebuffer;
	class Medium;
	class Renderer;
	class Scene;
	class IntegratorBase;
}
