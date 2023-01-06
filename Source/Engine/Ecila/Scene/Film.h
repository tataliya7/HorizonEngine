#pragma once

#include "EcilaCommon.h"
#include "EcilaMath.h"

namespace Ecila
{
	class Film
	{
	public:
		Film(uint32 width, uint32 height) : width(width), height(height) {}
		virtual ~Film() = default;

		Vector4u GetSampleBounds() const
		{
			return { 0, 0, width, height };
		}

	protected:

		uint32 width;
		uint32 height;
	};

	Vector3 ACESFilm(const Vector3& x)
	{
		static const Matrix3x3 mat1(Vector3(0.59719, 0.07600, 0.02840),
			glm::dvec3(0.35458, 0.90834, 0.13383),
			glm::dvec3(0.04823, 0.01566, 0.83777));

		static const Matrix3x3 mat2(Vector3(1.60475, -0.10208, -0.00327),
			Vector3(-0.53108, 1.10813, -0.07276),
			Vector3(-0.07367, -0.00605, 1.07602));

		auto RRTAndODTFit = [](const Vector3& v)
		{
			Vector3 a = v * (v + 0.0245786) - 0.000090537;
			Vector3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
			return a / b;
		};

		Vector3 color = mat2 * RRTAndODTFit(mat1 * x);

		return Math::Clamp(color, 0.0, 1.0);
	}
}
