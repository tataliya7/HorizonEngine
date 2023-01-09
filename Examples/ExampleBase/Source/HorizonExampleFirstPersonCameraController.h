#pragma once

#include "Core/CoreDefinitions.h"

import HorizonEngine.Core;
import HorizonEngine.Entity;

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

	protected:
		
		void UpdateTransform(float deltaTime, Vector3& outCameraPosition, Vector3& outCameraEuler);
	};
}