#pragma once

#include <vector>

import HorizonEngine.Core;

namespace HE
{
	struct PositionTrack
	{
		std::vector<float> times;
		std::vector<Vector3> positions;
	};

	struct RotationTrack
	{
		std::vector<float> times;
		std::vector<Vector3> rotations;
	};

	struct CurveTrack
	{
		std::vector<Vector3> controlPoints1;
		std::vector<Vector3> controlPoints2;
		void Reset()
		{
			controlPoints1.clear();
			controlPoints2.clear();
		}
	};

	class CameraAnimationSequence
	{
	public:

		CameraAnimationSequence() = default;
		~CameraAnimationSequence() = default;

		PositionTrack positionTrack;
		RotationTrack rotationTrack;

		CurveTrack positionCurve;
		CurveTrack rotationCurve;

		float speed = 1.0f;
		float currentTime = 0.0f;
		float timeLengthInSeconds = 0.0f;

		void Reset();

		void InitCurve();
		void AddPositionSample(float time, const Vector3& position);
		void AddRotationSample(float time, const Vector3& rotation);

		void UpdateCameraMotion(float deltaTime, Vector3& outPosition, Vector3& outRotation);

	private:
		void EvaluatePosition(float time, Vector3& outPosition);
		void EvaluateRotation(float time, Vector3& outRotation);
	};
}
