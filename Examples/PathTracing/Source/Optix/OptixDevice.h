#pragma once

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <string>
#include <sstream>
#include <exception>
#include <stdexcept>

class OptixDevice;
class OptixAccelerationStructure
{
public:
    OptixDevice* device;
    OptixTraversableHandle traversableHandle;
};

class Exception : public std::runtime_error
{
public:
    Exception(const char* msg)
        : std::runtime_error(msg)
    {
    }

    Exception(OptixResult res, const char* msg)
        : std::runtime_error(createMessage(res, msg).c_str())
    {
    }

private:
    std::string createMessage(OptixResult res, const char* msg)
    {
        std::ostringstream out;
        out << optixGetErrorName(res) << ": " << msg;
        return out.str();
    }
};

#define CUDA_CHECK(function) CudaCheck(function, #function, __FILE__, __LINE__ )

#define OPTIX_CHECK(function) OptixCheck(function, #function, __FILE__, __LINE__ )

inline void CudaCheck(cudaError_t error, const char* function, const char* file, unsigned int line)
{
    if (error != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA API: " << function << " failed with error: '" << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
        throw Exception(ss.str().c_str());
    }
}

inline void OptixCheck(OptixResult result, const char* function, const char* file, unsigned int line)
{
    if (result != OPTIX_SUCCESS)
    {
        std::stringstream ss;
        ss << "Optix API: " << function << " failed: " << file << ':' << line << ")\n";
        throw Exception(result, ss.str().c_str());
    }
}

bool OptixInit();

struct OptixDeviceCreateInfo
{

};

class OptixDevice
{
public:
    OptixDevice(const OptixDeviceCreateInfo& info);
    ~OptixDevice();
    OptixDeviceContext GetContext() const
    {
        return context;
    }
    CUstream GetStream() const
    {
        return stream;
    }
private:
    OptixDeviceContext context;
    CUstream stream;
};

OptixDevice* OptixDeviceCreate(const OptixDeviceCreateInfo& info);

void OptixDeviceDestroy(OptixDevice* device);