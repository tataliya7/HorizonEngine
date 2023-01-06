#pragma once

#include "EcilaCommon.h"
#include "EcilaMath.h"

namespace Ecila
{
	class Framebuffer
	{
	public:
		Framebuffer(uint32 width, uint32 height, const Vector3 & clearColor = { 0, 0, 0 });
		uint32 GetWidth() const { return mWidth; }
		uint32 GetHeight() const { return mHeight; }
		Vector3 GetClearColor() const { return mClearColor; }
		void SetClearColor(const Vector3& color) { mClearColor = color; }
		uint32 GetSize() const { return (uint32)mBuffer.size(); }
		Vector3 GetPixel(uint32 index) const { return mBuffer[index]; }
		Vector3 GetPixel(const Vector2i& pos) const { return GetPixel(pos.x, pos.y); }
		Vector3 GetPixel(uint32 i, uint32 j) const { return GetPixel(i * mWidth + j); }
		void SetPixel(uint32 index, const Vector3& color) { mBuffer[index] = color; }
		void SetPixel(const Vector2i& pos, const Vector3& color) { SetPixel(pos.x, pos.y, color); }
		void SetPixel(uint32 i, uint32 j, const Vector3& color) { SetPixel(i * mWidth + j, color); }
		void Accumulate(uint32 i, uint32 j, const Vector3& color) 
		{
			uint32 index = i * mWidth + j;
			mBuffer[index] += color;
		}
	private:
		uint32 mWidth;
		uint32 mHeight;
		Vector3 mClearColor;
		std::vector<Vector3> mBuffer;
	};
}
