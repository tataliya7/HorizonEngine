#pragma once

#include <HorizonEngine.h>

namespace HE
{

struct EditorCameraControllerUserImpulseData
{
	float moveRightLeftImpulse                = 0.0f;
	float moveUpDownImpulse                   = 0.0f;
	float moveForwardBackwardImpulse          = 0.0f;
	float rotatePitchImpulse                  = 0.0f;
	float rotateYawImpulse                    = 0.0f;
	float rotateRollImpulse                   = 0.0f;
	void Reset()
	{
		moveRightLeftImpulse                  = 0.0f;
		moveUpDownImpulse                     = 0.0f;
		moveForwardBackwardImpulse            = 0.0f;
		rotatePitchImpulse                    = 0.0f;
		rotateYawImpulse                      = 0.0f;
		rotateRollImpulse                     = 0.0f;
	}
};

struct EditorCameraControllerConfig
{
	float translationMultiplier               = 1.0f;
	float rotationMultiplier                  = 1.0f;
	float translationAccelerationRate         = 20000.0f;
	float rotationAccelerationRate            = 1600.0f;
	float maxTranslationVelocity              = FLOAT_MAX;
	float maxRotationVelocity                 = FLOAT_MAX;
	float minAllowedPitchRotation             = -90.0f;
	float maxAllowedPitchRotation             = 90.0f;
	float translationVelocityDampingAmount    = 20.0f;
	float rotationVelocityDampingAmount       = 25.0f;
	bool limitedPitch                         = true;
	bool usePhysicsallyBasedTranslation       = true;
	bool usePhysicsallyBasedRotation          = true;
};

class EditorCameraController
{
public:

	EditorCameraController();

	virtual ~EditorCameraController() = default;

	virtual void Update(const EditorCameraControllerUserImpulseData& userImpulseData, float deltaTime, float translationVelocityScale, Vector3& outCameraPosition, Vector3& outCameraEuler);

	bool IsRotating() const;

	/** Configuration. */
	EditorCameraControllerConfig config;

private:

	void UpdatePosition(const EditorCameraControllerUserImpulseData& userImpulseData, float deltaTime, float translationVelocityScale, const Vector3& cameraEuler, Vector3& outCameraPosition);

	void UpdateRotation(const EditorCameraControllerUserImpulseData& userImpulseData, float deltaTime, Vector3& outCameraEuler);

	/** World space translation velocity in centimeters per second. */
	Vector3 translationVelocity;

	/** Rotation velocity (pitch, yaw and roll) in degrees per second. */
	Vector3 rotationVelocityEuler;
};

}
