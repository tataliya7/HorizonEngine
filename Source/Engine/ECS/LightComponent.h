#pragma once

#include "ECS/ECSCommon.h"

namespace HE
{
	class LightRenderProxy;

	struct DirectionalLightComponent
	{
		Vector3 color;
		float intensity;
		DirectionalLightComponent()
		{
			using namespace entt;
			auto factory = entt::meta<DirectionalLightComponent>();
			factory.data<&DirectionalLightComponent::intensity, entt::as_ref_t>("Intensity"_hs)
				.prop("Name"_hs, std::string("Intensity"));
			factory.data<&DirectionalLightComponent::color, entt::as_ref_t>("Color"_hs)
				.prop("Name"_hs, std::string("Color"));
		}

		LightRenderProxy* proxy;
	};

	class SkyLightRenderProxy;

	struct SkyLightComponent
	{
		std::string cubemap;
		uint32 cubemapResolution;

		SkyLightComponent()
		{
			using namespace entt;
			auto factory = entt::meta<SkyLightComponent>();
			factory.data<&SkyLightComponent::cubemapResolution, entt::as_ref_t>("Cubemap Resolution"_hs)
				.prop("Name"_hs, std::string("Cubemap Resolution"));
			factory.data<&SkyLightComponent::cubemap, entt::as_ref_t>("Cubemap"_hs)
				.prop("Name"_hs, std::string("Cubemap"));
		}
		
		void SetCubemap(std::string newCubemap)
		{
			if (cubemap != newCubemap)
			{
				cubemap = newCubemap;
				SetDirty();
			}
		}

		void SetDirty()
		{
			valid = false;
		}

		SkyLightRenderProxy* proxy;

		bool valid;
	};
}
