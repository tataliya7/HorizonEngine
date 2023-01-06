#pragma once

#include "EcilaCommon.h"
#include "Optix/OptixDevice.h"
#include "Integrator/PathTracingIntegratorLaunchParams.h"

namespace Ecila
{
    class PathTracingIntegrator
    {
    public:
        PathTracingIntegrator(OptixDevice* device);
        ~PathTracingIntegrator();
        void Launch(const PathTracingIntegratorLaunchParams& params, uint32 width, uint32 height);
    protected:
        OptixDevice* device;
        
        OptixPipeline pipeline;
        OptixModule ptxModule;
        
        OptixProgramGroup raygenGroup;
        OptixProgramGroup missGroup;
        OptixProgramGroup shadowRayMissGroup;
        OptixProgramGroup primaryRayAndSecondaryRayHitGroup;
        OptixProgramGroup shadowRayHitGroup;
        
        OptixShaderBindingTable sbt;

        OptixTraversableHandle instanceAS;
    };
}
