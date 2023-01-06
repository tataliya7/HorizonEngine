#pragma once

struct LightSample
{
    float3 P;
    float3 Ng;
    float t;
    float u, v;
    float3 wi;
    float pdf;
    LightType type;
};

__forceinline__ __device__ void sample_rect_light(const KernelRectLight& rect_light, const float3& shading_point, LightSample& light_sample)
{
    const float area = length(cross(rect_light.v1, rect_light.v2));
    const float inv_area = 1.0f / area;

    light_sample.P = rect_light.corner + rect_light.v1 * light_sample.u + rect_light.v2 * light_sample.v;
    light_sample.Ng = rect_light.normal;
    light_sample.wi = normalize(light_sample.P - shading_point);
    light_sample.t = length(light_sample.P - shading_point);
    light_sample.pdf = inv_area;
}

__forceinline__ __device__ void sample_light(const KernelLight& light, const float u, const float v, const float3& shading_point, LightSample& light_sample)
{
    LightType light_type = light.type;
    light_sample.type = light_type;
    light_sample.u = u;
    light_sample.v = v;
    
    if (light_type == LIGHT_TYPE_RECT)
    {
        sample_rect_light(light.rect_light, shading_point, light_sample);
    }
}