#pragma once

#include "EcilaCommon.h"
#include "EcilaMath.h"
#include "Medium.h"
#include "Film.h"
#include "Ray.h"

namespace Ecila
{
	struct CameraRayGenerateInfo
	{

	};

	struct CameraSampleInfo
	{
		Vector2i pos;
		Vector2i lens;
		float time;
	};

	class Camera
	{
	public:
		Camera() = default;
		virtual ~Camera() = default;
		const Film& GetFilm() const { return *film; }
		void LookAt(Vector3 point)
		{
			forward = glm::normalize(point - position);
			left = glm::cross({ 0.0, 1.0, 0.0 }, forward);
			left = glm::length(left) < 1e-9 ? Vector3(-1.0f, 0.0f, 0.0f) : glm::normalize(left);
			up = glm::normalize(glm::cross(forward, left));
		}
	public:
		std::string name;
		Vector3 position;
		Vector3 forward, left, up;

		float focalLength;
		float focusDistance;
		float sensorWidth;
		float apertureRadius;

		Film* film;
	
	private:

	};

	struct CameraParameters
	{
		Matrix4 view = Matrix4(1);
		Matrix4 proj = Matrix4(1);
		Matrix4 viewProj = Matrix4(1);
		Matrix4 invViewProj = Matrix4(1);
		Vector3 posW = Vector3(0, 0, 0);
		float aspectRatio = 16.0f / 9.0f;
		Vector3 U = Vector3(1, 0, 0);
		float nearZ = 0.1f;
		Vector3 V = Vector3(0, 1, 0);
		float farZ = 1000.0f;
		Vector3 W = Vector3(0, 0, 1);
		float yFov = PI / 2.0f;
		Vector3 targetPoint = Vector3(0, 0, 1);
		float aperture = 0.0f;
		float focalLength = 1.0f;
		float focalDistance = 30.0f;
		float padding1;
		float padding2;
	};
}
