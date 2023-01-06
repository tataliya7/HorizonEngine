#include "OptixDevice.h"
#include <optix_function_table_definition.h>

#include <iostream>
#include <iomanip>

bool OptixInit()
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    return true;
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

OptixDevice::OptixDevice(const OptixDeviceCreateInfo& info)
{
    CUcontext cuContext = 0; // Zero means take the current context
    OptixDeviceContextOptions options = {};
    options.logCallbackLevel = 4; /* Fatal = 1, Error = 2, Warning = 3, Print = 4. */
    options.logCallbackFunction = &context_log_cb;
    //    [](unsigned int level, const char* tag, const char* message, void* cbdata)
    //{
    //    switch (level) 
    //    {
    //    case 1:
    //        //LOG_FATAL("{}", message);
    //        break;
    //    case 2:
    //        //LOG_ERROR("{}", message);
    //        break;
    //    case 3:
    //        //LOG_WARNING("{}", message);
    //        break;
    //    case 4:
    //        //LOG_INFO("{}", message);
    //        break;
    //    }
    //};
    OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &context));

    CUDA_CHECK(cudaStreamCreate(&stream));
}

OptixDevice::~OptixDevice()
{
    OPTIX_CHECK(optixDeviceContextDestroy(context));
}

OptixDevice* OptixDeviceCreate(const OptixDeviceCreateInfo& info)
{
    return new OptixDevice(info);
}

void OptixDeviceDestroy(OptixDevice* device)
{
    if (device != nullptr)
    {
        delete device;
    }
}
