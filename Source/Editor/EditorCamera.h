#pragma once

#include <HorizonEngine.h>

namespace HE
{
struct EditorCamera
{
	float fieldOfView;
	float aspectRatio;
	float zNear;
	float zFar;

	Vector3 position;
	Vector3 euler;

	Matrix4x4 viewMatrix;
	Matrix4x4 projectionMatrix;

	void Update();
};
}
