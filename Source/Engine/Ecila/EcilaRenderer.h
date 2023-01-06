#pragma once

import HorizonEngine.Core;
import HorizonEngine.Render;

namespace HE
{

	class DaisyRenderer : public Renderer
	{
	public:
		virtual void Render() override;
	private:
		void RenderScreenSpaceShadows();
		void RenderRayTracingShadows();
		void RenderAmbientOcclusion();
		RenderBackend* renderBackend;
		ShaderLibrary* shaderLibrary;
	};

}