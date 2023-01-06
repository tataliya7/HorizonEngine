#pragma once

#define GLM_FORCE_CTOR_INIT
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
// #define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/transform.hpp>

#include <chrono>

#include "StlHeaders.h"

#include <random>

namespace Ecila
{
    using Radian = float;
    using Degree = float;

    using Vector2i = glm::ivec2;
    using Vector3i = glm::ivec3;
    using Vector4i = glm::ivec4;

    using Vector2u = glm::uvec2;
    using Vector3u = glm::uvec3;
    using Vector4u = glm::uvec4;
    
    using Vector2 = glm::vec2;
    using Vector3 = glm::vec3;
    using Vector4 = glm::vec4;

    using Quaternion = glm::quat;

    using Matrix3 = glm::mat3;
    using Matrix4 = glm::mat4;

    const float PI = glm::pi<float>();

    const float EPSILON = 0.00001f;

    const Vector3 UNIT_X(1, 0, 0);
    const Vector3 UNIT_Y(0, 1, 0);
    const Vector3 UNIT_Z(0, 0, 1);
    
#define FLOAT_MAX			  (3.402823466e+38f)

    struct Rect
    {
        Vector2 minPoint;
        Vector2 maxPoint;
        Rect() : minPoint(0.0f, 0.0f), maxPoint(0.0f, 0.0f) {}
        Rect(const Vector2& min, const Vector2& max) : minPoint(min), maxPoint(max) {}
        Rect(float x1, float y1, float x2, float y2) : minPoint(x1, y1), maxPoint(x2, y2) {}
    };

    namespace Math
    {
        inline float Random()
        {
            static thread_local std::random_device dev;
            static thread_local std::mt19937 rng(dev());
            std::uniform_real_distribution<float> dist(0.f, 1.f);
            return dist(rng);
        }

        inline float Dot(const Vector3& a, const Vector3& b)
        {
            return glm::dot(a, b);
        }

        inline Vector3 Cross(const Vector3& a, const Vector3& b)
        {
            return glm::cross(a, b);
        }
    }

    namespace MathUtil
    {

        /* Returns a random real in [min,max). */
        inline float Random(float min, float max)
        {
            return min + (max - min) * Random();
        }

        inline static Vector3 RandomVector3(Real min = 0, Real max = 1)
        {
            return Vector3(Random(min, max), Random(min, max), Random(min, max));
        }

        inline Real Clamp(Real x, Real min = 0, Real max = 1)
        {
            if (x < min) return min;
            if (x > max) return max;
            return x;
        }

        inline Vector3 Clamp(const Vector3& vec, Real min = 0, Real max = 1)
        {
            return Vector3(Clamp(vec[0]), Clamp(vec[1]), Clamp(vec[2]));
        }

        inline Vector3 Min3(const Vector3& v1, const Vector3& v2)
        {
            return Vector3(std::min(v1[0], v2[0]), std::min(v1[1], v2[1]), std::min(v1[2], v2[2]));
        }

        inline Vector3 Max3(const Vector3& v1, const Vector3& v2)
        {
            return Vector3(std::max(v1[0], v2[0]), std::max(v1[1], v2[1]), std::max(v1[2], v2[2]));
        }

        inline float DegreeToRadians(Degree degree)
        {
            return degree * PI / 180;
        }

        inline Quaternion getQuatFromAxes(const Vector3& xAxis, const Vector3& yAxis, const Vector3& zAxis)
        {
            Matrix3 kRot;

            kRot[1][0] = xAxis.y;
            kRot[2][0] = xAxis.z;
            kRot[0][0] = xAxis.x;

            kRot[0][1] = yAxis.x;
            kRot[1][1] = yAxis.y;
            kRot[2][1] = yAxis.z;

            kRot[0][2] = zAxis.x;
            kRot[1][2] = zAxis.y;
            kRot[2][2] = zAxis.z;

            return glm::quat_cast(kRot);
        }
    }
}
