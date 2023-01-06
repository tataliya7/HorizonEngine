#include "EditorCamera.h"

namespace HE
{
    void EditorCamera::Update()
    {
        Quaternion orientation = Quaternion(Math::DegreesToRadians(euler));
        viewMatrix = Math::Inverse(Math::Compose(position, orientation, Vector3(1.0f, 1.0f, 1.0f)));
        projectionMatrix = glm::perspectiveLH(Math::DegreesToRadians(fieldOfView), aspectRatio, zNear, zFar);
    }
}