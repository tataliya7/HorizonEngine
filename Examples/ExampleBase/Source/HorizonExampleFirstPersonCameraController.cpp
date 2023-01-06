#include "HorizonExampleFirstPersonCameraController.h"

import HorizonEngine.Input;

namespace HE
{
    void HorizonExampleFirstPersonCameraController::UpdateTransform(float deltaTime, Vector3& outCameraPosition, Vector3& outCameraEuler)
    {
        float delta = 5.0;

        float moveForwardBackward = 0.0f;
        float moveRightLeft = 0.0f;
        float moveUpDown = 0.0f;

        float rotatePitch = 0.0f;
        float rotateYaw = 0.0f;
        float rotateRoll = 0.0f;

        if (Input::GetKeyDown(KeyCode::W))
        {
            moveForwardBackward += delta;
        }
        if (Input::GetKeyDown(KeyCode::S))
        {
            moveForwardBackward -= delta;
        }
        if (Input::GetKeyDown(KeyCode::D))
        {
            moveRightLeft += delta;
        }
        if (Input::GetKeyDown(KeyCode::A))
        {
            moveRightLeft -= delta;
        }
        if (Input::GetKeyDown(KeyCode::E))
        {
            moveUpDown += delta;
        }
        if (Input::GetKeyDown(KeyCode::Q))
        {
            moveUpDown -= delta;
        }

        static Vector2 lastMousePos = { 0.0f, 0.0f };
        Vector2 mousePos;
        Input::GetMousePosition(mousePos.x, mousePos.y);
        Vector2 mouseMovement = (mousePos - lastMousePos);
        lastMousePos = mousePos;

        if (Input::GetMouseButtonDown(MouseButtonID::ButtonMiddle))
        {
            moveRightLeft -= mouseMovement.x * translationMultiplier;
            moveUpDown += mouseMovement.y * translationMultiplier;
        }
        else if (Input::GetMouseButtonDown(MouseButtonID::ButtonRight))
        {
            rotatePitch -= mouseMovement.y * rotationMultiplier;
            rotateYaw -= mouseMovement.x * rotationMultiplier;
        }

        const float cameraBoost = Input::GetKeyDown(KeyCode::LeftShift) ? 2.0f : 1.0f;
        const float finalCameraSpeed = cameraSpeed * cameraBoost;

        Vector3 localSpaceTranslationVelocity = Vector3(
            moveRightLeft,         // pitch
            moveForwardBackward,   // roll
            moveUpDown             // yaw
        );

        const Quaternion cameraOrientation = Quaternion(Math::DegreesToRadians(outCameraEuler));
        Vector3 worldSpaceTranslationVelocity = cameraOrientation * localSpaceTranslationVelocity;
        Vector3 translationVelocity = worldSpaceTranslationVelocity * finalCameraSpeed;

        if (Math::LengthSquared(translationVelocity) > Math::Square(maxTranslationVelocity * finalCameraSpeed))
        {
            translationVelocity = Math::Normalize(translationVelocity) * maxTranslationVelocity * finalCameraSpeed;
        }
        if (Math::LengthSquared(translationVelocity) < Math::Square(KINDA_SMALL_NUMBER))
        {
            translationVelocity = Vector3(0.0f, 0.0f, 0.0f);
        }

        outCameraPosition += translationVelocity * deltaTime;

        Vector3 rotationVelocityEuler = Vector3(
            rotatePitch,    // pitch
            rotateRoll,     // roll
            rotateYaw       // yaw
        );

        for (uint32 eulerAxis = 0; eulerAxis < 3; eulerAxis++)
        {
            float rotationVelocity = rotationVelocityEuler[eulerAxis];

            rotationVelocity = Math::Clamp(rotationVelocity, -maxRotationVelocity, maxRotationVelocity);
            if (Math::Abs(rotationVelocity) < KINDA_SMALL_NUMBER)
            {
                rotationVelocity = 0.0f;
            }

            outCameraEuler[eulerAxis] += rotationVelocity * deltaTime;

            // Limit final pitch rotation value to configured range.
            if (eulerAxis == 1)
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

                outCameraEuler[eulerAxis] = Math::Clamp(eulerAngle, -90.0f, 90.0f);
            }
        }
    }
}