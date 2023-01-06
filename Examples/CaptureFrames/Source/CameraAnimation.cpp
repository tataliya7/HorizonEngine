#include "CameraAnimation.h"

namespace HE
{
	static void UpdateControlPoints(const std::vector<Vector3>& knot, std::vector<Vector3>& outCtrl1, std::vector<Vector3>& outCtrl2)
	{
		uint32 numPoints = (knot.empty()) ? 0 : (uint32)knot.size();
		if (numPoints > 1)
		{
			outCtrl1.resize(numPoints);
			outCtrl2.resize(numPoints);

			// Compute smooth control points
			{
				if (numPoints <= 2)
				{
					if (numPoints == 2)
					{
						outCtrl1[0] = Math::Lerp(knot[0], knot[1], 0.33333f);
						outCtrl2[0] = Math::Lerp(knot[0], knot[1], 0.66666f);
					}
					else if (numPoints == 1)
					{
						outCtrl1[0] = outCtrl2[0] = knot[0];
					}
					return;
				}

				std::vector<float> a(numPoints);
				std::vector<float> b(numPoints);
				std::vector<float> c(numPoints);
				std::vector<float> r(numPoints);
				for (uint32 axis = 0; axis < 3; axis++)
				{
					int n = numPoints - 1;

					a[0] = 0;
					b[0] = 2;
					c[0] = 1;
					r[0] = knot[0][axis] + 2 * knot[1][axis];

					for (int i = 1; i < n - 1; i++)
					{
						a[i] = 1;
						b[i] = 4;
						c[i] = 1;
						r[i] = 4 * knot[i][axis] + 2 * knot[i + 1][axis];
					}

					a[n - 1] = 2;
					b[n - 1] = 7;
					c[n - 1] = 0;
					r[n - 1] = 8 * knot[n - 1][axis] + knot[n][axis];

					for (int i = 1; i < n; i++)
					{
						float m = a[i] / b[i - 1];
						b[i] = b[i] - m * c[i - 1];
						r[i] = r[i] - m * r[i - 1];
					}

					outCtrl1[n - 1][axis] = r[n - 1] / b[n - 1];
					for (int i = n - 2; i >= 0; i--)
					{
						outCtrl1[i][axis] = (r[i] - c[i] * outCtrl1[i + 1][axis]) / b[i];
					}

					for (int i = 0; i < n; i++)
					{
						outCtrl2[i][axis] = 2 * knot[i + 1][axis] - outCtrl1[i + 1][axis];
					}
					outCtrl2[n - 1][axis] = 0.5f * (knot[n][axis] + outCtrl1[n - 1][axis]);
				}
			}
		}
	}

	void CameraAnimationSequence::Reset()
	{
		speed = 1.0f;
		currentTime = 0.0f;
		timeLengthInSeconds = 0.0f;
		positionTrack.times.clear();
		positionTrack.positions.clear();
		rotationTrack.times.clear();
		rotationTrack.rotations.clear();
		positionCurve.Reset();
		rotationCurve.Reset();
	}

	void CameraAnimationSequence::InitCurve()
	{
		UpdateControlPoints(positionTrack.positions, positionCurve.controlPoints1, positionCurve.controlPoints2);
		UpdateControlPoints(rotationTrack.rotations, rotationCurve.controlPoints1, rotationCurve.controlPoints2);
	}

	void CameraAnimationSequence::AddPositionSample(float time, const Vector3& position)
	{
		positionTrack.times.push_back(time);
		positionTrack.positions.push_back(position);
	}

	void CameraAnimationSequence::AddRotationSample(float time, const Vector3& rotation)
	{
		rotationTrack.times.push_back(time);
		rotationTrack.rotations.push_back(rotation);
	}

	void CameraAnimationSequence::UpdateCameraMotion(float deltaTime, Vector3& outPosition, Vector3& outRotation)
	{
		currentTime += deltaTime * speed;
		if (currentTime > timeLengthInSeconds)
		{
			currentTime = 0.0f;
		}
		EvaluatePosition(currentTime, outPosition);
		EvaluateRotation(currentTime, outRotation);
	}

	void CameraAnimationSequence::EvaluatePosition(float time, Vector3& outPosition)
	{
		if (positionTrack.times.empty())
		{
			return;
		}

		uint32 index1 = 0, index2 = 0;
		for (uint32 i = 0; i < (uint32)positionTrack.times.size(); i++)
		{
			if (time >= positionTrack.times[i] && time <= positionTrack.times[i + 1])
			{
				index1 = i;
				index2 = i + 1;
				break;
			}
		}

		if (index1 == index2)
		{
			outPosition = positionTrack.positions[index1];
		}
		else
		{
			float t = (time - positionTrack.times[index1]) / (positionTrack.times[index2] - positionTrack.times[index1]);
			outPosition = Math::Bezier3(
				t,
				positionTrack.positions[index1],
				positionCurve.controlPoints1[index1],
				positionCurve.controlPoints2[index1],
				positionTrack.positions[index2]
			);
		}
	}

	void CameraAnimationSequence::EvaluateRotation(float time, Vector3& outRotation)
	{
		if (rotationTrack.times.empty())
		{
			return;
		}

		uint32 index1 = 0, index2 = 0;
		for (uint32 i = 0; i < (uint32)rotationTrack.times.size(); i++)
		{
			if (time >= rotationTrack.times[i] && time <= rotationTrack.times[i + 1])
			{
				index1 = i;
				index2 = i + 1;
				break;
			}
		}

		if (index1 == index2)
		{
			outRotation = rotationTrack.rotations[index1];
		}
		else
		{
			float t = (time - rotationTrack.times[index1]) / (rotationTrack.times[index2] - rotationTrack.times[index1]);
			outRotation = Math::Bezier3(
				t,
				rotationTrack.rotations[index1],
				rotationCurve.controlPoints1[index1],
				rotationCurve.controlPoints2[index1],
				rotationTrack.rotations[index2]
			);
		}
	}
}