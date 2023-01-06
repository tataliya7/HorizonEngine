#pragma once

static __forceinline__ __device__ void to_unit_disk(float& x, float& y)
{
    float phi = M_TWO_PI_F * x;
    float r = sqrtf(y);
    x = r * cosf(phi);
    y = r * sinf(phi);
}

static __forceinline__ __device__ void sample_uniform_hemisphere(const float3& N, float u, float v, float3& wi, float& pdf)
{
    float r = sqrtf(max(0.0f, 1.0f - u * u));
    float phi = M_TWO_PI_F * v;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = u;
    
    float3 T, B;
    make_orthonormals(N, T, B);
    
    wi = x * T + y * B + z * N;
    pdf = 0.5f * M_INV_PI_F;
}

static __forceinline__ __device__ void sample_cosine_hemisphere(const float3& N, float u, float v, float3& wi, float& pdf)
{
    to_unit_disk(u, v);
    float costheta = sqrtf(max(0.0f, 1.0f - u * u - v * v));
    
    float3 T, B;
    make_orthonormals(N, T, B);
    
    wi = u * T + v * B + costheta * N;
    pdf = costheta * M_INV_PI_F;
}
