module;

#include <PxPhysicsAPI.h>

#include "Core/CoreDefinitions.h"

export module HorizonEngine.Physics.Core;

import HorizonEngine.Core;

#define PHYSX_PVD_HOST "127.0.0.1"

export namespace HE
{
	physx::PxDefaultAllocator        GPhysXAllocator;
	physx::PxDefaultErrorCallback    GPhysXErrorCallback;
	physx::PxDefaultCpuDispatcher*   GPhysXCpuDispatcher = nullptr;
	physx::PxFoundation*             GPhysXFoundation = nullptr;
	physx::PxPvd*                    GPhysXPvd = nullptr;
	physx::PxPhysics*                GPhysXSDK = nullptr;

	bool PhysXInit()
	{
		// Do nothing if SDK already exists
		if (GPhysXFoundation != nullptr)
		{
			return true;
		}

		GPhysXFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, GPhysXAllocator, GPhysXErrorCallback);
		ASSERT(GPhysXFoundation != nullptr);

		GPhysXPvd = physx::PxCreatePvd(*GPhysXFoundation);
		ASSERT(GPhysXPvd != nullptr);
		physx::PxPvdTransport* transport = physx::PxDefaultPvdSocketTransportCreate(PHYSX_PVD_HOST, 5425, 10);
		GPhysXPvd->connect(*transport, physx::PxPvdInstrumentationFlag::eALL);

		GPhysXSDK = PxCreatePhysics(PX_PHYSICS_VERSION, *GPhysXFoundation, physx::PxTolerancesScale(), true, GPhysXPvd);
		ASSERT(GPhysXSDK != nullptr);

		GPhysXCpuDispatcher = physx::PxDefaultCpuDispatcherCreate(2);
		ASSERT(GPhysXCpuDispatcher != nullptr);

		return true;
	}

	void PhysXExit()
	{
		PX_RELEASE(GPhysXCpuDispatcher);
		PX_RELEASE(GPhysXSDK);
		if (GPhysXPvd)
		{
			physx::PxPvdTransport* transport = GPhysXPvd->getTransport();
			PX_RELEASE(GPhysXPvd);
			PX_RELEASE(transport);
		}
		PX_RELEASE(GPhysXFoundation);
	}
}