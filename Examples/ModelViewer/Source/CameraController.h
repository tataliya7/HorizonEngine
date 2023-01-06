#pragma once

#include <HorizonEngine.h>

class SimpleFirstPersonCameraController
{
public:

	SimpleFirstPersonCameraController() = default;

	~SimpleFirstPersonCameraController() = default;

	void Update(float deltaTime, HE::Vector3& outCameraPosition, HE::Vector3& outCameraEuler);

	float cameraSpeed = 1.0f;

	float maxTranslationVelocity = FLOAT_MAX;

	float maxRotationVelocity = FLOAT_MAX;

	float translationMultiplier = 10.0f;

	float rotationMultiplier = 10.0f;
};