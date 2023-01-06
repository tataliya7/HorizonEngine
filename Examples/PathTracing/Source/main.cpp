#include "EcilaCommon.h"
#include "EcilaMath.h"
#include "Optix/OptixDevice.h"
#include "Integrator/PathTracingIntegrator.h"
#include "Integrator/PathTracingIntegratorLaunchParams.h"

#include <array>
#include <vector>
#include <fstream>
#include <iostream>

namespace Ecila
{
    inline size_t AlignUp(size_t offset, size_t alignment)
    {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    struct Image
    {
        unsigned int width = 0;
        unsigned int height = 0;
        void* data = nullptr;
    };

    static bool SaveAsPPM(const char* filename, const unsigned char* data, int width, int height, int channels)
    {
        if (data == nullptr || width < 1 || height < 1)
        {
            return false;
        }

        if (channels != 1 && channels != 3 && channels != 4)
        {
            return false;
        }

        std::ofstream stream(filename, std::ios::out | std::ios::binary);
        if (!stream.is_open())
        {
            return false;
        }

        bool is_float = false;
        stream << 'P';
        stream << ((channels == 1 ? (is_float ? 'Z' : '5') : (channels == 3 ? (is_float ? '7' : '6') : '8'))) << std::endl;
        stream << width << " " << height << std::endl << 255 << std::endl;

        stream.write(reinterpret_cast<char*>(const_cast<unsigned char*>(data)), width * height * channels * (is_float ? 4 : 1));
        stream.close();

        return true;
    }

    void BuildAccelerationStructure(OptixDevice* device, const OptixBuildInput& buildInput, OptixAccelerationStructure* accelerationSturcture)
    {
        bool preferFastTrace = true;
        OptixAccelBuildOptions buildOptions = {};
        buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        if (preferFastTrace)
        {
            buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        }
        else
        {
            buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        }

        OptixAccelBufferSizes bufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            device->GetContext(),
            &buildOptions,
            &buildInput,
            1, 
            &bufferSizes));

        CUdeviceptr tempBuffer;
        CUDA_CHECK(cudaMalloc((void**)&tempBuffer, AlignUp(bufferSizes.tempSizeInBytes, 8) + 8));

        CUdeviceptr outputBuffer;
        CUDA_CHECK(cudaMalloc((void**)&outputBuffer, bufferSizes.outputSizeInBytes));

        OptixAccelEmitDesc compactedSizeProperty = {};
        compactedSizeProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        compactedSizeProperty.result             = (CUdeviceptr)((char*)tempBuffer + AlignUp(bufferSizes.tempSizeInBytes, 8));

        OptixTraversableHandle traversableHandle = 0;
        OPTIX_CHECK(optixAccelBuild(
            device->GetContext(), 
            0, 
            &buildOptions,
            &buildInput, 
            1, 
            tempBuffer, 
            bufferSizes.tempSizeInBytes, 
            outputBuffer, 
            bufferSizes.outputSizeInBytes, 
            &traversableHandle, 
            preferFastTrace ? &compactedSizeProperty : nullptr, 
            preferFastTrace ? 1 : 0));

        CUDA_CHECK(cudaStreamSynchronize(0));

        if (preferFastTrace)
        {
            uint64 compactedSize;
            CUDA_CHECK(cudaMemcpy(&compactedSize, (void*)compactedSizeProperty.result, sizeof(compactedSize), cudaMemcpyDeviceToHost));

            if (compactedSize < bufferSizes.outputSizeInBytes)
            {
                CUdeviceptr compactedBufffer;
                CUDA_CHECK(cudaMalloc((void**)&compactedBufffer, compactedSize));

                OPTIX_CHECK(optixAccelCompact(
                    device->GetContext(), 
                    0, 
                    traversableHandle, 
                    compactedBufffer, 
                    compactedSize, 
                    &traversableHandle));

                CUDA_CHECK(cudaStreamSynchronize(0));

                CUDA_CHECK(cudaFree((void*)outputBuffer));
                outputBuffer = compactedBufffer;
            }
        }

        CUDA_CHECK(cudaFree((void*)tempBuffer));

        accelerationSturcture->traversableHandle = traversableHandle;
    }

    void RenderScene(OptixDevice* device)
    {
        uint32 width = 768;//1280;
        uint32 height = 768;//720;

        uint32 numVertices = 96;
        const std::array<float3, 96> vertices = {{
            // Floor  -- white lambert
            {    0.0f,    0.0f,    0.0f },
            {    0.0f,    0.0f,  559.2f },
            {  556.0f,    0.0f,  559.2f },
            {    0.0f,    0.0f,    0.0f },
            {  556.0f,    0.0f,  559.2f },
            {  556.0f,    0.0f,    0.0f },

            // Ceiling -- white lambert
            {    0.0f,  548.8f,    0.0f },
            {  556.0f,  548.8f,    0.0f },
            {  556.0f,  548.8f,  559.2f },

            {    0.0f,  548.8f,    0.0f },
            {  556.0f,  548.8f,  559.2f },
            {    0.0f,  548.8f,  559.2f },

            // Back wall -- white lambe
            {    0.0f,    0.0f,  559.2f },
            {    0.0f,  548.8f,  559.2f },
            {  556.0f,  548.8f,  559.2f },

            {    0.0f,    0.0f,  559.2f },
            {  556.0f,  548.8f,  559.2f },
            {  556.0f,    0.0f,  559.2f },

            // Right wall -- green lamb
            {    0.0f,    0.0f,    0.0f },
            {    0.0f,  548.8f,    0.0f },
            {    0.0f,  548.8f,  559.2f },

            {    0.0f,    0.0f,    0.0f },
            {    0.0f,  548.8f,  559.2f },
            {    0.0f,    0.0f,  559.2f },

            // Left wall -- red lambert
            {  556.0f,    0.0f,    0.0f },
            {  556.0f,    0.0f,  559.2f },
            {  556.0f,  548.8f,  559.2f },

            {  556.0f,    0.0f,    0.0f },
            {  556.0f,  548.8f,  559.2f },
            {  556.0f,  548.8f,    0.0f },

            // Short block -- white lam
            {  130.0f,  165.0f,   65.0f },
            {   82.0f,  165.0f,  225.0f },
            {  242.0f,  165.0f,  274.0f },

            {  130.0f,  165.0f,   65.0f },
            {  242.0f,  165.0f,  274.0f },
            {  290.0f,  165.0f,  114.0f },

            {  290.0f,    0.0f,  114.0f },
            {  290.0f,  165.0f,  114.0f },
            {  240.0f,  165.0f,  272.0f },

            {  290.0f,    0.0f,  114.0f },
            {  240.0f,  165.0f,  272.0f },
            {  240.0f,    0.0f,  272.0f },

            {  130.0f,    0.0f,   65.0f },
            {  130.0f,  165.0f,   65.0f },
            {  290.0f,  165.0f,  114.0f },

            {  130.0f,    0.0f,   65.0f },
            {  290.0f,  165.0f,  114.0f },
            {  290.0f,    0.0f,  114.0f },

            {   82.0f,    0.0f,  225.0f },
            {   82.0f,  165.0f,  225.0f },
            {  130.0f,  165.0f,   65.0f },

            {   82.0f,    0.0f,  225.0f },
            {  130.0f,  165.0f,   65.0f },
            {  130.0f,    0.0f,   65.0f },

            {  240.0f,    0.0f,  272.0f },
            {  240.0f,  165.0f,  272.0f },
            {   82.0f,  165.0f,  225.0f },

            {  240.0f,    0.0f,  272.0f },
            {   82.0f,  165.0f,  225.0f },
            {   82.0f,    0.0f,  225.0f },

            // Tall block -- white lamb
            {  423.0f,  330.0f,  247.0f },
            {  265.0f,  330.0f,  296.0f },
            {  314.0f,  330.0f,  455.0f },

            {  423.0f,  330.0f,  247.0f },
            {  314.0f,  330.0f,  455.0f },
            {  472.0f,  330.0f,  406.0f },

            {  423.0f,    0.0f,  247.0f },
            {  423.0f,  330.0f,  247.0f },
            {  472.0f,  330.0f,  406.0f },

            {  423.0f,    0.0f,  247.0f },
            {  472.0f,  330.0f,  406.0f },
            {  472.0f,    0.0f,  406.0f },

            {  472.0f,    0.0f,  406.0f },
            {  472.0f,  330.0f,  406.0f },
            {  314.0f,  330.0f,  456.0f },

            {  472.0f,    0.0f,  406.0f },
            {  314.0f,  330.0f,  456.0f },
            {  314.0f,    0.0f,  456.0f },

            {  314.0f,    0.0f,  456.0f },
            {  314.0f,  330.0f,  456.0f },
            {  265.0f,  330.0f,  296.0f },

            {  314.0f,    0.0f,  456.0f },
            {  265.0f,  330.0f,  296.0f },
            {  265.0f,    0.0f,  296.0f },

            {  265.0f,    0.0f,  296.0f },
            {  265.0f,  330.0f,  296.0f },
            {  423.0f,  330.0f,  247.0f },

            {  265.0f,    0.0f,  296.0f },
            {  423.0f,  330.0f,  247.0f },
            {  423.0f,    0.0f,  247.0f },

            // Ceiling light -- emmissi
            {  343.0f,  548.6f,  227.0f },
            {  213.0f,  548.6f,  227.0f },
            {  213.0f,  548.6f,  332.0f },

            {  343.0f,  548.6f,  227.0f },
            {  213.0f,  548.6f,  332.0f },
            {  343.0f,  548.6f,  332.0f }
        }};

        std::array<uint32_t, 32> g_mat_indices = {{
            0, 0,                          // Floor         -- white lambert
            0, 0,                          // Ceiling       -- white lambert
            0, 0,                          // Back wall     -- white lambert
            1, 1,                          // Right wall    -- green lambert
            2, 2,                          // Left wall     -- red lambert
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Short block   -- white lambert
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Tall block    -- white lambert
            3, 3                           // Ceiling light -- emmissive
        }};

        CUdeviceptr  d_mat_indices             = 0;
        const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof( uint32_t );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_mat_indices ), mat_indices_size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( d_mat_indices ),
                    g_mat_indices.data(),
                    mat_indices_size_in_bytes,
                    cudaMemcpyHostToDevice));

        CUdeviceptr vertexBuffer;
        const size_t vertices_size_in_bytes = vertices.size() * sizeof(float3);
        CUDA_CHECK(cudaMalloc((void**)&vertexBuffer, vertices_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(
            (void*)vertexBuffer,
            vertices.data(), vertices_size_in_bytes,
            cudaMemcpyHostToDevice));

        uint32_t triangle_input_flags[4] =  // One per SBT record for this build input
        {
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
        };

        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
        buildInput.triangleArray.numVertices = numVertices;
        buildInput.triangleArray.vertexBuffers = &vertexBuffer;
        buildInput.triangleArray.numSbtRecords = 4;
        buildInput.triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
        buildInput.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32);
        buildInput.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32);
        buildInput.triangleArray.flags = triangle_input_flags;

        OptixAccelerationStructure* tlas = new OptixAccelerationStructure(device);
        BuildAccelerationStructure(device, buildInput, tlas);

        CUDA_CHECK(cudaFree((void*)d_mat_indices));

        uchar4* device_pixels = nullptr;
        uint64 size = width * height * sizeof(uchar4);
        CUDA_CHECK(cudaMalloc(&device_pixels, size));

        PathTracingIntegratorLaunchParams launchParams = {};
        launchParams.frame_buffer = device_pixels;
        launchParams.frame_buffer_width = width;
        launchParams.frame_buffer_height = height;
        launchParams.num_samples = 128;
        launchParams.tlas = tlas->traversableHandle;
        launchParams.vertices = (float3*)vertexBuffer;
        launchParams.light.type = LIGHT_TYPE_RECT;
        launchParams.light.emission = make_float3( 15.0f, 15.0f, 5.0f );
        launchParams.light.rect_light.corner   = make_float3( 343.0f, 548.5f, 227.0f );
        launchParams.light.rect_light.v1       = make_float3( 0.0f, 0.0f, 105.0f );
        launchParams.light.rect_light.v2       = make_float3( -130.0f, 0.0f, 0.0f );
        launchParams.light.rect_light.normal   = normalize(cross(launchParams.light.rect_light.v1, launchParams.light.rect_light.v2 ) );

        PathTracingIntegrator integrator(device);
        integrator.Launch(launchParams, width, height);

        std::vector<uchar4> host_pixels(width * height);

        CUDA_CHECK(cudaMemcpy(
            host_pixels.data(),
            device_pixels,
            width * height * sizeof(uchar4),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaStreamSynchronize(device->GetStream()));

        Image image;
        image.width = width;
        image.width = height;
        image.data = host_pixels.data();

        std::vector<unsigned char> pixels(width * height * 3);
        {
            for(int j = height - 1; j >= 0; j--)
            {
                for(int i = 0; i < width; i++)
                {
                    const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                    const int32_t src_idx = 4*width*j            + 4*i;
                    pixels[dst_idx + 0] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+0 ];
                    pixels[dst_idx + 1] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+1 ];
                    pixels[dst_idx + 2] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+2 ];
                }
            }
        }

        SaveAsPPM("D:/test.ppm", pixels.data(), width, height, 3);
    }

    int EcilaMain(int argc, const char** argv)
    {
        if (!OptixInit())
        {
            return -1;
        }

        OptixDeviceCreateInfo deviceInfo = {};
        OptixDevice* device = OptixDeviceCreate(deviceInfo);

        RenderScene(device);

        OptixDeviceDestroy(device);

        return 0;
    }
}

int main(int argc, const char** argv)
{
    int exitCode = 0;
    
    try
    {
        exitCode = Ecila::EcilaMain(argc, argv);
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return exitCode;
}
