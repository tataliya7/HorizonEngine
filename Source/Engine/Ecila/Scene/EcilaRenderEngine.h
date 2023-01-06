#pragma once

#include "EcilaCommon.h"

namespace Ecila
{
	class Scene;
	class Camera;
	class ThreadPool;
	class Integrator;

	class EcilaRenderEngine
	{
	public:
		EcilaRenderEngine();
		~EcilaRenderEngine();
		void SetSPP(uint32 spp)
		{
			samplesPerPixel = spp;
		}
		void RenderOneFrame(Scene* scene, Camera* camera, Framebuffer* framebuffer);
	protected:
		uint32 frameID;
		uint32 tileSize;
		uint32 numThreads;
		uint32 samplesPerPixel;
		ThreadPool* threadPool;
		Integrator* integrator;
	};
}
