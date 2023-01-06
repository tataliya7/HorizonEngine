#include "EditorCameraController.h"

namespace HE
{

EditorCameraController::EditorCameraController()
	: config()
	, translationVelocity(0.0f, 0.0f, 0.0f)
	, rotationVelocityEuler(0.0f, 0.0f, 0.0f)
{

}

bool EditorCameraController::IsRotating() const
{
	if ((rotationVelocityEuler.x != 0.0f) || (rotationVelocityEuler.y != 0.0f) || (rotationVelocityEuler.z != 0.0f))
	{
		return true;
	}
	return false;
}

void EditorCameraController::Update(const EditorCameraControllerUserImpulseData& userImpulseData, float deltaTime, float translationVelocityScale, Vector3& outCameraPosition, Vector3& outCameraEuler)
{
	// Translation
	UpdatePosition(userImpulseData, deltaTime, translationVelocityScale, outCameraEuler, outCameraPosition);

	// Rotation
	UpdateRotation(userImpulseData, deltaTime, outCameraEuler);
}

void EditorCameraController::UpdatePosition(const EditorCameraControllerUserImpulseData& userImpulseData, float deltaTime, float translationVelocityScale, const Vector3& cameraEuler, Vector3& outCameraPosition)
{
	Vector3 localSpaceTranslationImpulse = Vector3(
		userImpulseData.moveRightLeftImpulse,         // pitch
		userImpulseData.moveUpDownImpulse,            // yaw
		userImpulseData.moveForwardBackwardImpulse    // roll
	);

	Vector3 worldSpaceTranslationAcceleration;
	{
		// Compute camera orientation, then rotate the local space translation impulse to world space.
		const Quaternion cameraOrientation = Quaternion(Math::DegreesToRadians(cameraEuler));
		Vector3 worldSpaceTranslationImpulse = cameraOrientation * localSpaceTranslationImpulse;
		worldSpaceTranslationAcceleration = worldSpaceTranslationImpulse * config.translationAccelerationRate * translationVelocityScale;
	}

	if (config.usePhysicsallyBasedTranslation)
	{
		// Accelerate the movement velocity
		translationVelocity += worldSpaceTranslationAcceleration * deltaTime;

		// Apply damping
		{
			const float dampingFactor = Math::Clamp(config.translationVelocityDampingAmount * deltaTime, 0.0f, 0.75f);
			translationVelocity += -translationVelocity * dampingFactor;
		}
	}
	else
	{
		translationVelocity = worldSpaceTranslationAcceleration;
	}

	// Clamp
	if (Math::LengthSquared(translationVelocity) > Math::Square(config.maxTranslationVelocity * translationVelocityScale))
	{
		translationVelocity = Math::Normalize(translationVelocity) * config.maxTranslationVelocity * translationVelocityScale;
	}
	if (Math::LengthSquared(translationVelocity) < Math::Square(KINDA_SMALL_NUMBER))
	{
		translationVelocity = Vector3(0.0f, 0.0f, 0.0f);
	}

	outCameraPosition += translationVelocity * deltaTime;
}

void EditorCameraController::UpdateRotation(const EditorCameraControllerUserImpulseData& userImpulseData, float deltaTime, Vector3& outCameraEuler)
{
	Vector3 rotationImpulseEuler = Vector3(
		userImpulseData.rotatePitchImpulse,    // pitch
		userImpulseData.rotateYawImpulse,      // yaw
		userImpulseData.rotateRollImpulse      // roll
	);

	// Iterate for each euler axis (pitch, yaw and roll).
	for (uint32 eulerAxis = 0; eulerAxis < 3; eulerAxis++)
	{
		const float rotationImpulse = rotationImpulseEuler[eulerAxis];

		float rotationAcceleration = rotationImpulse * config.rotationAccelerationRate;

		float& rotationVelocity = rotationVelocityEuler[eulerAxis];
		if (config.usePhysicsallyBasedRotation)
		{
			// Accelerate the rotation velocity
			rotationVelocity += rotationAcceleration * deltaTime;

			// Apply damping
			{
				const float dampingFactor = Math::Clamp(config.rotationVelocityDampingAmount * deltaTime, 0.0f, 0.75f);
				rotationVelocity += -rotationVelocity * dampingFactor;
			}
		}
		else
		{
			rotationVelocity = rotationAcceleration;
		}

		// Clamp
		rotationVelocity = Math::Clamp(rotationVelocity, -config.maxRotationVelocity, config.maxRotationVelocity);
		if (Math::Abs(rotationVelocity) < KINDA_SMALL_NUMBER)
		{
			rotationVelocity = 0.0f;
		}

		outCameraEuler[eulerAxis] += rotationVelocity * deltaTime;

		// Limit final pitch rotation value to configured range.
		if (eulerAxis == 0)
		{
			// Normalize the angle to -180 to 180.
			float eulerAngle = Math::Fmod(outCameraEuler[eulerAxis], 360.0f);
			if (eulerAngle > 180.f)
			{
				eulerAngle -= 360.f;
			}
			else if (eulerAngle < -180.f)
			{
				eulerAngle += 360.f;
			}

			if (config.limitedPitch)
			{
				outCameraEuler[eulerAxis] = Math::Clamp(eulerAngle, config.minAllowedPitchRotation, config.maxAllowedPitchRotation);
			}
		}
	}
}

}