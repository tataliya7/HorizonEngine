#pragma once

struct PathTracingIntegratorRayPayload
{
    float3                 ray_origin;
    float3                 ray_direction;
    float3                 light_eval;
    float3                 emitted;
    float3                 throughput;
    unsigned int           seed;
    unsigned int           primary_ray;
    unsigned int           done;
};

enum LightType
{
    LIGHT_TYPE_DIRECTION   = 0,
    LIGHT_TYPE_POINT       = 1,
    LIGHT_TYPE_SPOT        = 2,
    LIGHT_TYPE_RECT        = 3,
    LIGHT_TYPE_ENVIRONMENT = 4,
};

struct KernelRectLight
{
    float3 corner;
    float3 v1;
    float3 v2;
    float3 normal;
};

struct KernelLight
{
    LightType type;
    float3 emission;
    float max_bounces;
    union 
    {
        KernelRectLight rect_light;
    };
};

struct KernelCamera
{
    float near_plane;
    float far_plane;
    float sensor_width;
    float sensor_height;
};

struct KernelFilm
{
    float exposure;
};

template <typename T>
struct OptixSbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RaygenRecordData
{

};

struct MissRecordData
{

};

struct HitgroupRecordData
{
    float3  emission_color;
    float3  diffuse_color;
};

using RaygenRecord = OptixSbtRecord<RaygenRecordData>;
using MissRecord = OptixSbtRecord<MissRecordData>;
using HitgroupRecord = OptixSbtRecord<HitgroupRecordData>;

struct PathTracingIntegratorLaunchParams
{
    uchar4* frame_buffer;
    unsigned int frame_buffer_width;
    unsigned int frame_buffer_height;
    unsigned int num_samples;
    unsigned int padding;
    OptixTraversableHandle tlas;
    float3* vertices;
    KernelCamera camera;
    KernelFilm film;
    KernelLight light;
};
