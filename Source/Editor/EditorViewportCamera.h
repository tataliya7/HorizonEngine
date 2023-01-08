#pragma once

#include <HorizonEngine.h>

namespace HE
{
	namespace Editor
	{
		void SetViewportCameraPosition(AzFramework::ViewportId viewportId, const Vector3& position);
		void SetViewportCameraRotation(AzFramework::ViewportId viewportId, const Vector3& rotation);
		void SetViewportCameraTransform(AzFramework::ViewportId viewportId, const Vector3& position, const Vector3& rotation);
	}
}
