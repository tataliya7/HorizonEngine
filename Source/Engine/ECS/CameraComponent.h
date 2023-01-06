#pragma once

#include "ECS/ECSCommon.h"

namespace HE
{
	enum class CameraType
	{
		Perpective,
		Orthographic,
	};

	struct CameraComponent
	{
		CameraType type;
		float nearPlane;
		float farPlane;
		float fieldOfView;
		float aspectRatio;
		bool overrideAspectRatio;

		CameraComponent()
		{
			using namespace entt;
			auto factory = entt::meta<CameraComponent>();
			factory.data<&CameraComponent::overrideAspectRatio, entt::as_ref_t>("Override Aspect Ratio"_hs)
				.prop("Name"_hs, std::string("Override Aspect Ratio"));
			factory.data<&CameraComponent::aspectRatio, entt::as_ref_t>("Aspect Ratio"_hs)
				.prop("Name"_hs, std::string("Aspect Ratio"));
			factory.data<&CameraComponent::fieldOfView, entt::as_ref_t>("Field of View"_hs)
				.prop("Name"_hs, std::string("Field of View"));
			factory.data<&CameraComponent::farPlane, entt::as_ref_t>("Far Plane"_hs)
				.prop("Name"_hs, std::string("Far Plane"));
			factory.data<&CameraComponent::nearPlane, entt::as_ref_t>("Near Plane"_hs)
				.prop("Name"_hs, std::string("Near Plane"));
		}
	};
	//void UpdateCameraViewMatrix(CameraComponent* camera, const Transform& transform);
	//void UpdateCameraProjectionMatrix(CameraComponent* camera);
}
