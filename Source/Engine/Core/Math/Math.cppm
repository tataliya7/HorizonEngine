module;

#include "CoreCommon.h"
#include <chrono>

export module HorizonEngine.Core.Math;

export import "MathDefinitions.h";

import HorizonEngine.Core.Types;

export namespace HE
{
    using Vector2 = glm::vec2;
    using Vector3 = glm::vec3;
    using Vector4 = glm::vec4;

    using Vector2i = glm::ivec2;
    using Vector3i = glm::ivec3;
    using Vector4i = glm::ivec4;

    using Vector2u = glm::uvec2;
    using Vector3u = glm::uvec3;
    using Vector4u = glm::uvec4;

    using Vector2f = glm::fvec2;
    using Vector3f = glm::fvec3;
    using Vector4f = glm::fvec4;

    using Vector2d = glm::dvec2;
    using Vector3d = glm::dvec3;
    using Vector4d = glm::dvec4;

    using Matrix3x3 = glm::mat3;
    using Matrix4x4 = glm::mat4;

    using Quaternion = glm::quat;

    struct Transform
    {
        Vector4 translation;
        Vector4 rotation;
        Vector4 scale;
    };

    struct Rect
    {
        Vector2 minimum;
        Vector2 maximum;

        Rect() : minimum(0.0f, 0.0f), maximum(0.0f, 0.0f) {}
        Rect(float x0, float y0, float x1, float y1) : minimum(x0, y0), maximum(x1, y1) {}
        Rect(Vector2 min, Vector2 max) : minimum(min), maximum(max) {}

        float GetWidth() const { return maximum.x - minimum.x; }
        float GetHeight() const { return maximum.y - minimum.y; }
        Vector2 GetSize() const { return Vector2(maximum.x - minimum.x, maximum.y - minimum.y); }
    };
}

export namespace HE::Math
{
    FORCEINLINE float Halton(int32 index, int32 base)
    {
        float result = 0.0f;
        float invBase = 1.0f / base;
        float fraction = invBase;
        while (index > 0)
        {
            result += (index % base) * fraction;
            index /= base;
            fraction *= invBase;
        }
        return result;
    }

    FORCEINLINE bool IsPowerOfTwo(uint32 n)
    {
        return (n > 0) && ((n & (n - 1)) == 0);
    }

    FORCEINLINE bool IsPowerOfTwo(int32 n)
    {
        return (n > 0) && ((n & (n - 1)) == 0);
    }

    FORCEINLINE double Cos(double radians)
    {
        return cos(radians);
    }

    FORCEINLINE float Cos(float radians)
    {
        return cos(radians);
    }

    template <typename T>
    FORCEINLINE T Max(const T& a, const T& b)
    {
        return (a >= b) ? a : b;
    }

    template <typename T>
    FORCEINLINE T Min(const T& a, const T& b)
    {
        return (a <= b) ? a : b;
    }

    FORCEINLINE float Lerp(float x, float y, float t)
    {
        return x + (y - x) * t;
    }

    FORCEINLINE Vector3 Lerp(const Vector3& x, const Vector3& y, float t)
    {
        return x + (y - x) * t;
    }

    FORCEINLINE float Abs(float x)
    {
        return abs(x);
    }

    FORCEINLINE float Fmod(float x, float y)
    {
        return fmod(x, y);
    }

    FORCEINLINE float Square(float x)
    {
        return x * x;
    }

    FORCEINLINE Vector3 Normalize(const Vector3& v)
    {
        return glm::normalize(v);
    }

    FORCEINLINE float Length(const Vector3& v)
    {
        return glm::length(v);
    }

    FORCEINLINE float LengthSquared(const Vector3& v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }

    FORCEINLINE Matrix4x4 Transpose(const Matrix4x4& matrix)
    {
        return glm::transpose(matrix);
    }

    FORCEINLINE Matrix4x4 Compose(const Vector3& translation, const Quaternion& rotation, const Vector3& scale)
    {
        return glm::translate(glm::mat4(1), translation) * glm::mat4_cast(glm::normalize(rotation)) * glm::scale(glm::mat4(1.0f), scale);
    }

    FORCEINLINE void Decompose(const Matrix4x4& matrix, Vector3& outTranslation, Quaternion& outRotation, Vector3& outScale)
    {
        Vector3 skew;
        Vector4 perspective;
        glm::decompose(matrix, outScale, outRotation, outTranslation, skew, perspective);
    }

    FORCEINLINE uint32 MaxNumMipLevels(uint32 size)
    {
        return 1 + uint32(std::floor(std::log2(size)));
    }

    FORCEINLINE uint32 MaxNumMipLevels(uint32 width, uint32 height)
    {
        return 1 + uint32(std::floor(std::log2(glm::min(width, height))));
    }

    FORCEINLINE float Clamp(float x, float min = 0, float max = 1)
    {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    FORCEINLINE uint32 Clamp(uint32 x, uint32 min = 0, uint32 max = 1)
    {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    FORCEINLINE Vector3 Clamp(const Vector3& vec, float min = 0, float max = 1)
    {
        return Vector3(Clamp(vec[0]), Clamp(vec[1]), Clamp(vec[2]));
    }

    FORCEINLINE Vector3 Bezier3(float t, Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3)
    {
        t = Math::Clamp(t, 0.0f, 1.0f);
        float d = 1.0f - t;
        return d * d * d * p0 + 3.0f * d * d * t * p1 + 3.0f * d * t * t * p2 + t * t * t * p3;
    }

    FORCEINLINE float DegreesToRadians(float x)
    {
        return glm::radians(x);
    }

    FORCEINLINE Vector3 DegreesToRadians(const Vector3& v)
    {
        return glm::radians(v);
    }

    FORCEINLINE float RadiansToDegrees(float x)
    {
        return glm::radians(x);
    }

    FORCEINLINE Vector3 EulerAnglesFromQuat(const Quaternion& quat)
    {
        return glm::eulerAngles(quat);
    }

    FORCEINLINE Quaternion QuaternionFromAngleAxis(float angle, const Vector3& axis)
    {
        return glm::angleAxis(angle, axis);
    }

    FORCEINLINE Matrix4x4 Inverse(const Matrix4x4& matrix)
    {
        return glm::inverse(matrix);
    }

    FORCEINLINE Matrix4x4 PerspectiveReverseZ_RH_ZO(float fovy, float aspect, float zNear, float zFar)
    {
        return glm::perspectiveRH_ZO(fovy, aspect, zFar, zNear);
    };
}
