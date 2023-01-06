#include "EcilaRenderEngine.h"
#include "EcilaMath.h"
#include "Camera.h"
#include "Framebuffer.h"
#include "ThreadPool.h"
#include "Integrator.h"

namespace Ecila
{
	static void SaveFramebufferToFile(const char* filename, Framebuffer* framebuffer)
	{
		FILE* fp = fopen(filename, "wb");
		fprintf(fp, "P6\n%d %d\n255\n", framebuffer->GetWidth(), framebuffer->GetHeight());
		for (uint32 i = 0; i < framebuffer->GetSize(); i++)
		{
			static unsigned char color[3];
			Vector3 pixel = framebuffer->GetPixel(i);
			color[0] = (unsigned char)(255 * Math::Clamp(std::pow(Math::Clamp(pixel[0]), 0.6f)));
			color[1] = (unsigned char)(255 * Math::Clamp(std::pow(Math::Clamp(pixel[1]), 0.6f)));
			color[2] = (unsigned char)(255 * Math::Clamp(std::pow(Math::Clamp(pixel[2]), 0.6f)));
			fwrite(color, 1, 3, fp);
		}
		fclose(fp);
	}

	EcilaRenderEngine::EcilaRenderEngine()
		: frameID(0)
		, tileSize(16)
		, numThreads(16)
		, samplesPerPixel(512)
		, threadPool(nullptr)
		, integrator(nullptr)
	{
		threadPool = new ThreadPool(numThreads);
		integrator = new PathTracingIntegrator();
	}

	EcilaRenderEngine::~EcilaRenderEngine()
	{
		delete threadPool;
		delete integrator;
	}

    void EcilaRenderEngine::RenderOneFrame(Scene* scene, Camera* camera, Framebuffer* framebuffer)
    {
		Vector4u sampleBounds = camera->GetFilm().GetSampleBounds();
		Vector2u sampleExtent = { sampleBounds.z - sampleBounds.x, sampleBounds.w - sampleBounds.y };
		Vector2u numTiles = { (sampleExtent.x + tileSize - 1) / tileSize, (sampleExtent.y + tileSize - 1) / tileSize };

		for (uint32 sampleIndex = 0; sampleIndex < samplesPerPixel; sampleIndex++)
		{
			for (uint32 x = 0; x < numTiles.x; x++)
			{
				for (uint32 y = 0; y < numTiles.y; y++)
				{
					Vector2 tilePos = { x, y };
					threadPool->Execute([=]()
					{
						uint32 x0 = sampleBounds.x + tilePos.x * tileSize;
						uint32 x1 = std::min(x0 + tileSize, sampleBounds.z);
						uint32 y0 = sampleBounds.y + tilePos.y * tileSize;
						uint32 y1 = std::min(y0 + tileSize, sampleBounds.w);

						for (uint32 i = x0; i < x1; i++)
						{
							for (uint32 j = y0; j < y1; j++)
							{
								Sampler::initiate(static_cast<uint32_t>(i * sampleExtent.x + j));
								Sampler::setIndex(sampleIndex);
								Vector2 subpixelJitter = Sampler::get<0, 2>();
								Vector2 pixelCoord = { i, j };
								Vector2 uv = pixelCoord + subpixelJitter;
								Vector2 halfExtent = Vector2(sampleExtent.x, sampleExtent.y) * 0.5f;
								float pixel_size = sensor_width / image.width;
								Vector2 localOffset = pixel_size * (halfExtent - uv);
								Vector3 rayOrigin = camera->position;
								Vector3 rayDirection = glm::normalize(camera->forward * camera->focalLength + camera->left * localOffset.x + camera->up * localOffset.y);
								Ray cameraRay(rayOrigin, rayDirection);

								Vector3 L = integrator->Li(cameraRay, scene);
								framebuffer->Accumulate(i, j, L);
							}
						}
					});
				}
			}

			// Save framebuffer to file
			SaveFramebufferToFile("binary.ppm", framebuffer);
		}

        frameID++;
    }
}