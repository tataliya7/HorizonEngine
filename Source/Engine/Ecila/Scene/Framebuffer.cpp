#include "Framebuffer.h"

namespace Ecila
{
	SharedPtr<Framebuffer> Framebuffer::Create(uint32 width, uint32 height, const Vector3& clearColor)
	{
		return SharedPtr<Framebuffer>(new Framebuffer(width, height, clearColor));
	}

	Framebuffer::Framebuffer(uint32 width, uint32 height, const Vector3& clearColor)
		: mWidth(width)
		, mHeight(height)
		, mClearColor(clearColor)
		, mBuffer(width * height, clearColor)
	{
	}
}
