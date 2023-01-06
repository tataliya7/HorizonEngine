#pragma once

#include "EcilaDefinitions.h"

#include <vector_functions.h>
#include <vector_types.h>

#if !defined(__CUDACC_RTC__)
#include <cmath>
#include <cstdlib>
#endif

#define M_PI_F         3.1415926535897932f
#define M_TWO_PI_F     6.2831853071795864f
#define M_HALF_PI_F    1.5707963267948966f
#define M_INV_PI_F     0.3183098861837067f

typedef unsigned char          uchar;
typedef unsigned short         ushort;
typedef unsigned int           uint;
typedef long long int          longlong;
typedef unsigned long long int ulonglong;

/* float2 */
static FORCEINLINE CUDA_HOST_DEVICE float2 make_float2(const float a)
{
    return make_float2(a, a);
}

static FORCEINLINE CUDA_HOST_DEVICE float2 operator+(const float2& a, const float2& b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

static FORCEINLINE CUDA_HOST_DEVICE void operator+=(float2& a, const float2& b)
{
    a.x += b.x; a.y += b.y;
}

static FORCEINLINE CUDA_HOST_DEVICE float2 operator-(const float2& a)
{
    return make_float2(-a.x, -a.y);
}

static FORCEINLINE CUDA_HOST_DEVICE float2 operator-(const float2& a, const float2& b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

static FORCEINLINE CUDA_HOST_DEVICE float2 operator-(const float2& a, const float b)
{
    return make_float2(a.x - b, a.y - b);
}

static FORCEINLINE CUDA_HOST_DEVICE float2 operator*(const float2& a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

static FORCEINLINE CUDA_HOST_DEVICE float2 operator*(float a, const float2& b)
{
    return b * a;
}

static FORCEINLINE CUDA_HOST_DEVICE float2 operator*(const float2& a, const float2& b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

static FORCEINLINE CUDA_HOST_DEVICE void operator*=(float2& a, const float2& b)
{
    a.x *= b.x; a.y *= b.y;
}

static FORCEINLINE CUDA_HOST_DEVICE float2 operator/(const float2& a, float b)
{
    return a * (1.0f / b);
}

/* float3 */
static FORCEINLINE CUDA_HOST_DEVICE float3 make_float3(const float a)
{
    return make_float3(a, a, a);
}

static FORCEINLINE CUDA_HOST_DEVICE float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static FORCEINLINE CUDA_HOST_DEVICE void operator+=(float3& a, const float3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

static FORCEINLINE CUDA_HOST_DEVICE float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

static FORCEINLINE CUDA_HOST_DEVICE float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static FORCEINLINE CUDA_HOST_DEVICE float3 operator-(const float3& a, const float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}

static FORCEINLINE CUDA_HOST_DEVICE float3 operator*(const float3& a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

static FORCEINLINE CUDA_HOST_DEVICE float3 operator*(float a, const float3& b)
{
    return b * a;
}

static FORCEINLINE CUDA_HOST_DEVICE float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static FORCEINLINE CUDA_HOST_DEVICE void operator*=(float3& a, const float3& b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

static FORCEINLINE CUDA_HOST_DEVICE float3 operator/(const float3& a, float b)
{
    return a * (1.0f / b);
}

static FORCEINLINE CUDA_HOST_DEVICE float3 cross(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static FORCEINLINE CUDA_HOST_DEVICE float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static FORCEINLINE CUDA_HOST_DEVICE float length(const float3& a)
{
    return sqrtf(dot(a, a));
}

static FORCEINLINE CUDA_HOST_DEVICE float3 normalize(const float3& a)
{
    return a / length(a);
}

/* float4 */
static FORCEINLINE CUDA_HOST_DEVICE float4 make_float4(const float a)
{
    return make_float4(a, a, a, a);
}

static FORCEINLINE CUDA_HOST_DEVICE float4 operator+(const float4& a, const float4& b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static FORCEINLINE CUDA_HOST_DEVICE void operator+=(float4& a, const float4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

static FORCEINLINE CUDA_HOST_DEVICE float4 operator-(const float4& a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

static FORCEINLINE CUDA_HOST_DEVICE float4 operator-(const float4& a, const float4& b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

static FORCEINLINE CUDA_HOST_DEVICE float4 operator-(const float4& a, const float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}

static FORCEINLINE CUDA_HOST_DEVICE float4 operator*(const float4& a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

static FORCEINLINE CUDA_HOST_DEVICE float4 operator*(float a, const float4& b)
{
    return b * a;
}

static FORCEINLINE CUDA_HOST_DEVICE float4 operator*(const float4& a, const float4& b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

static FORCEINLINE CUDA_HOST_DEVICE void operator*=(float4& a, const float4& b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}

static FORCEINLINE CUDA_HOST_DEVICE float4 operator/(const float4& a, float b)
{
    return a * (1.0f / b);
}

static FORCEINLINE CUDA_HOST_DEVICE float clamp(float v, float a, float b)
{
    return fmaxf(a, fminf(v, b));
}

static FORCEINLINE CUDA_HOST_DEVICE float2 clamp(const float2& v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

static FORCEINLINE CUDA_HOST_DEVICE float3 clamp(const float3& v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

static FORCEINLINE CUDA_HOST_DEVICE float4 clamp(const float4& v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

static FORCEINLINE CUDA_HOST_DEVICE void make_orthonormals(const float3& N, float3& T, float3& B)
{
    if (N.x != N.y || N.x != N.z)
    {
        T = make_float3(N.z - N.y, N.x - N.z, N.y - N.x); // (1, 1, 1) x N
    }
    else
    {
        T = make_float3(N.z - N.y, N.x + N.z, -N.y - N.x); // (-1, 1, 1) x N
    }
    T = normalize(T);
    B = normalize(cross(N, T));
}

static FORCEINLINE CUDA_HOST_DEVICE float3 faceforward(const float3& n, const float3& i, const float3& nref)
{
    return -n * copysignf(1.0f, dot(i, nref));
}