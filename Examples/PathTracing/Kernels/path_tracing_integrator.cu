#include <optix.h>
#include <optix_device.h>

#include "EcilaMath.h"
#include "Integrator/PathTracingIntegratorLaunchParams.h"

#include "common.h"
#include "random.h"
#include "sampling.h"
#include "light.h"
#include "color.h"

template<typename T> 
static __forceinline__ __device__ T *get_payload_ptr_0()
{
    return reinterpret_cast<T*>(unpack_payload_ptr(optixGetPayload_0(), optixGetPayload_1()));
}

template<typename T> 
static __forceinline__ __device__ T *get_payload_ptr_1()
{
    return reinterpret_cast<T*>(unpack_payload_ptr(optixGetPayload_2(), optixGetPayload_3()));
}

extern "C" static __constant__ PathTracingIntegratorLaunchParams launch_params;

extern "C" __global__ void __closesthit__path_tracing_integrator_shadow_ray()
{
    optixSetPayload_0(0);
}

extern "C" __global__ void __closesthit__path_tracing_integrator()
{
    HitgroupRecordData* rt_data = (HitgroupRecordData*)optixGetSbtDataPointer();
    PathTracingIntegratorRayPayload* payload = get_payload_ptr_0<PathTracingIntegratorRayPayload>();

    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float tmax = optixGetRayTmax();

    const int primitive_index = optixGetPrimitiveIndex();

    const unsigned int vertex_index_offset = primitive_index * 3;
    const float3 v0 = launch_params.vertices[vertex_index_offset + 0];
    const float3 v1 = launch_params.vertices[vertex_index_offset + 1];
    const float3 v2 = launch_params.vertices[vertex_index_offset + 2];
    
    const float3 Ng = normalize(cross(v1 - v0, v2 - v0));
    const float3 N = faceforward(Ng, ray_direction, Ng);

    const float3 shading_point = ray_origin + tmax * ray_direction;

    if (payload->primary_ray == 1)
    {
        payload->emitted = rt_data->emission_color;
        payload->primary_ray = 0;
    }
    else
    {
        payload->emitted = make_float3(0.0f);
    }

    // BSDF bsdf = eval_bsdf();

    unsigned int seed = payload->seed;
    {
        const float randu = randf(seed);
        const float randv = randf(seed);

        float3 wi;
        float pdf;
        sample_uniform_hemisphere(N, randu, randv, wi, pdf);
        
        // Secondary ray
        payload->ray_origin    = shading_point;
        payload->ray_direction = wi;
        payload->throughput   *= rt_data->diffuse_color;
    }

    const float randu = randf(seed);
    const float randv = randf(seed);
    
    const KernelLight& light = launch_params.light;

    LightSample light_sample;
    sample_light(light, randu, randv, shading_point, light_sample);

    float3 light_eval = light.emission;

    const float distance = light_sample.t;
    const float cos_theta = dot(N, light_sample.wi);
    const float cos_phi   = dot(light_sample.Ng, -light_sample.wi);

    float weight = 0.0f;
    if (cos_theta > 0.0f && cos_phi > 0.0f)
    {
        const float3 shadow_ray_origin = shading_point;
        const float3 shadow_ray_direction = light_sample.wi;
        const float shadow_ray_tmin = 0.01f;
        const float shadow_ray_tmax = light_sample.t - 0.01f;

        unsigned int visibility = 1;
        optixTrace(
            launch_params.tlas,
            shadow_ray_origin,
            shadow_ray_direction,
            shadow_ray_tmin,
            shadow_ray_tmax,
            0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            1, // SBToffset
            2, // SBTstride
            1, // missSBTIndex
            visibility);

        if (visibility == 1)
        {
            weight = cos_theta * cos_phi / (M_PI_F * distance * distance * light_sample.pdf);
        }
    }

    payload->light_eval = light_eval * weight;
}

extern "C" __global__ void __miss__path_tracing_integrator()
{
    PathTracingIntegratorRayPayload* payload = get_payload_ptr_0<PathTracingIntegratorRayPayload>();
    payload->light_eval = make_float3(0.0f);
    payload->done = 1;
}

extern "C" __global__ void __raygen__path_tracing_integrator_cast_ray()
{
    const uint3 launch_index = optixGetLaunchIndex();
    const unsigned int width = launch_params.frame_buffer_width;
    const unsigned int height = launch_params.frame_buffer_height;
    const unsigned int num_samples = launch_params.num_samples;
    const unsigned int max_num_bounces = 8;

    // TODO:
    const float3 camera_position = make_float3( 278.0f, 273.0f, -900.0f );
    float3 U = make_float3(-387.817566f, 0.0f, 0.0f);
    float3 V = make_float3( 0.0f, -387.817566f, 0.0f );
    float3 W = make_float3( 0.0f, 0.0f, 1230.0f );

    unsigned int seed = tea(launch_index.y * width + launch_index.x, 0);

    float3 accumulated_color = make_float3(0.0f);
    for (unsigned int sample_index = 0; sample_index < num_samples; sample_index++)
    {
        const float2 subpixel_jitter = make_float2(randf(seed), randf(seed));
        float2 uv = make_float2(((float)launch_index.x + subpixel_jitter.x) / (float)width, 
                                             ((float)launch_index.y + subpixel_jitter.y) / (float)height);
        uv.y = 1.0f - uv.y;
        uv = 2.0f * uv - 1.0f;

        // Camera ray
        float3 ray_origin = camera_position;
        float3 ray_direction = normalize(uv.x * U + uv.y * V + W);
        
        const float tmin = 0.01f;
        const float tmax = 1e16f;
        
        PathTracingIntegratorRayPayload payload = {};
        payload.light_eval  = make_float3(0.0f);
        payload.throughput  = make_float3(1.0f);
        payload.seed        = seed;
        payload.primary_ray = 1;
        payload.done        = 0;
        
        unsigned int depth = 0;

        for (;;)
        {
            unsigned int u0, u1;
            pack_payload_ptr(&payload, u0, u1);

            optixTrace(
                launch_params.tlas,
                ray_origin, 
                ray_direction, 
                tmin, 
                tmax, 
                0.0f, 
                OptixVisibilityMask(1), 
                OPTIX_RAY_FLAG_NONE, 
                0, // SBToffset
                2, // SBTstride
                0, // missSBTIndex
                u0, 
                u1);

            float3 L = payload.emitted + payload.light_eval * payload.throughput;
            accumulated_color += L;

            depth++;

            // TODO: Russian roulette
            if (payload.done == 1 || depth >= max_num_bounces)
            {
                break;
            }

            ray_origin    = payload.ray_origin;
            ray_direction = payload.ray_direction;
        }
    }

    float3 result = accumulated_color / (float)launch_params.num_samples;
    launch_params.frame_buffer[launch_index.y * launch_params.frame_buffer_width + launch_index.x] = make_color(result);
}
