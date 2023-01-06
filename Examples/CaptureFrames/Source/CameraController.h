#pragma once

#include "ECS/ECS.h"
#include "CameraAnimation.h"

import HorizonEngine.Core;

class SimpleFirstPersonCameraController : public HE::Scriptable
{
public:

	SimpleFirstPersonCameraController() = default;

	virtual ~SimpleFirstPersonCameraController() = default;

	HE::CameraAnimationSequence cameraAnim;
	
	float cameraSpeed = 1.0f;

	float maxTranslationVelocity = FLOAT_MAX;

	float maxRotationVelocity = FLOAT_MAX;

	float translationMultiplier = 10.0f;

	float rotationMultiplier = 10.0f;

	void OnCreate()
	{
		cameraAnim.Reset();
		cameraAnim.timeLengthInSeconds = 30.0f;

		cameraAnim.AddPositionSample(0.0f, HE::Vector3(-10.123f, 4.507f, 5.226f));
		cameraAnim.AddRotationSample(0.0f, HE::Vector3(-11.149f, 0.000f, -155.316f));

		cameraAnim.AddPositionSample(5.0f, HE::Vector3(-9.651f, -0.032f, 4.861f));
		cameraAnim.AddRotationSample(5.0f, HE::Vector3(-4.552f, 0.000f, -161.695f));

		cameraAnim.AddPositionSample(10.0f, HE::Vector3(-7.590f, -2.993f, 4.607f));
		cameraAnim.AddRotationSample(10.0f, HE::Vector3(-0.625f, 0.000f, -107.049f));

		cameraAnim.AddPositionSample(15.0f, HE::Vector3(-4.391f, -1.828f, 4.995f));
		cameraAnim.AddRotationSample(15.0f, HE::Vector3(-5.293f, 0.000f, -46.153f));

		cameraAnim.AddPositionSample(20.0f, HE::Vector3(-1.339f, 0.214f, 4.020f));
		cameraAnim.AddRotationSample(20.0f, HE::Vector3(-31.321f, 0.000f, -68.891f));

		cameraAnim.AddPositionSample(25.0f, HE::Vector3(1.773f, 0.486f, 2.424f));
		cameraAnim.AddRotationSample(25.0f, HE::Vector3(-21.487f, 0.000f, -90.298f));

		cameraAnim.AddPositionSample(30.0f, HE::Vector3(6.927f, 0.278f, 1.341f));
		cameraAnim.AddRotationSample(30.0f, HE::Vector3(0.703f, 0.000f, -90.552f));

		cameraAnim.InitCurve();
	}

	void OnDestroy()
	{

	}

	void OnUpdate(float deltaTime)
	{
		auto& tranform = GetComponent<HE::TransformComponent>();

		////if (Input::GetKeyDown(KeyCode::R))
		////{
		////	printf("cameraAnim.AddPositionSample(%.3f, HE::Vector3(%.3ff, %.3ff, %.3ff));\n", 0.0f, tranform.position.x, tranform.position.y, tranform.position.z);
		////	printf("cameraAnim.AddPositionSample(%.3f, HE::Vector3(%.3ff, %.3ff, %.3ff));\n", 0.0f, tranform.rotation.x, tranform.rotation.y, tranform.rotation.z);
		////	printf("\n");
		////	//HE_LOG_INFO("cameraAnim.AddPositionSample({:.3f}, HE::Vector3({:.3f}, {:.3f}, {:.3f}))", 0.0f, tranform.position.x, tranform.position.y, tranform.position.z);
		//////	HE_LOG_INFO("cameraAnim.AddRotationSample({:.3f}, HE::Vector3({:.3f}, {:.3f}, {:.3f}))", 0.0f, tranform.rotation.x, tranform.rotation.y, tranform.rotation.z);
		////}

		if (false)
		{
			float targetFPS = 30.0f;
			cameraAnim.UpdateCameraMotion(1.0f / targetFPS, tranform.position, tranform.rotation);
		}
		else
		{
			Update(deltaTime, tranform.position, tranform.rotation);
		}
	}

	void Update(float deltaTime, HE::Vector3& outCameraPosition, HE::Vector3& outCameraEuler);
};