export module HorizonEngine.Core.Types;

export namespace HE
{
	using int8 = signed char;
	using int16 = short;
	using int32 = int;
	using int64 = long long;

	using uint8 = unsigned char;
	using uint16 = unsigned short;
	using uint32 = unsigned int;
	using uint64 = unsigned long long;

	using char8 = uint8;
	using char16 = uint16;
	using char32 = uint32;

	using wchar = wchar_t;

	struct Extent2D
	{
		uint32 width;
		uint32 height;
	};

	struct Extent3D
	{
		uint32 width;
		uint32 height;
		uint32 depth;
	};

	struct Offset2D
	{
		int32 x;
		int32 y;
	};

	struct Offset3D
	{
		int32 x;
		int32 y;
		int32 z;
	};
}
