#pragma once

#include "ECS/ECS.h"

import HorizonEngine.Core;

namespace HE
{
	class HorizonExampleFirstPersonCameraController : public Scriptable
	{
	public:

		HorizonExampleFirstPersonCameraController() = default;
		virtual ~HorizonExampleFirstPersonCameraController() = default;

		float cameraSpeed = 1.0f;
		float maxTranslationVelocity = FLOAT_MAX;
		float maxRotationVelocity = FLOAT_MAX;
		float translationMultiplier = 10.0f;
		float rotationMultiplier = 10.0f;

		void OnCreate()
		{
			
		}

		void OnDestroy()
		{

		}

		void OnUpdate(float deltaTime)
		{
			auto& tranform = GetComponent<TransformComponent>();
			UpdateTransform(deltaTime, tranform.position, tranform.rotation);
		}

	private:
		
		void UpdateTransform(float deltaTime, Vector3& outCameraPosition, Vector3& outCameraEuler);
	};
}