module;

#include <PxPhysicsAPI.h>
#include <ECS/ECS.h>

export module HorizonEngine.Physics;

import HorizonEngine.Core;

#define PHYSX_PVD_HOST "127.0.0.1"

namespace HE::PhysXUtils 
{
	physx::PxTransform ToPhysXTransform(const Matrix4x4& transform)
	{
		Quaternion quat = glm::quat_cast(transform);
		physx::PxVec3 position(transform[3][0], transform[3][1], transform[3][2]);
		physx::PxQuat rotation(quat.x, quat.y, quat.z, quat.w);
		return physx::PxTransform(position, rotation);
	}

	physx::PxVec3 ToPhysXVec3(const Vector3& v)
	{
		return physx::PxVec3(v.x, v.y, v.z);
	}

	physx::PxVec4 ToPhysXVec4(const Vector4& v)
	{
		return physx::PxVec4(v.x, v.y, v.z, v.w);
	}

	physx::PxQuat ToPhysXQuat(const Quaternion& q)
	{
		return physx::PxQuat(q.x, q.y, q.z, q.w);
	}

	Matrix4x4 FromPhysXMatrix(const physx::PxMat44& matrix)
	{ 
		return Matrix4x4(
			matrix.column0.x, matrix.column1.x, matrix.column2.x, matrix.column3.x,
			matrix.column0.y, matrix.column1.y, matrix.column2.y, matrix.column3.y,
			matrix.column0.z, matrix.column1.z, matrix.column2.z, matrix.column3.z,
			matrix.column0.w, matrix.column1.w, matrix.column2.w, matrix.column3.w);
	}

	Vector3 FromPhysXVector(const physx::PxVec3& v)
	{ 
		return Vector3(v.x, v.y, v.z);
	}

	Vector4 FromPhysXVector(const physx::PxVec4& v)
	{ 
		return Vector4(v.x, v.y, v.z, v.w);
	}

	Quaternion FromPhysXQuat(const physx::PxQuat& q)
	{ 
		return Quaternion(q.x, q.y, q.z, q.w);
	}
}

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

	class PhysicsActor
	{
	public:
		void SynchronizeTransform()
		{
			TransformComponent& transform = manager->GetComponent<TransformComponent>(entity);
			manager->AddOrReplaceComponent<TransformDirtyComponent>(entity);

			physx::PxTransform pose = physxActor->getGlobalPose();
			
			transform.position = PhysXUtils::FromPhysXVector(pose.p);
			transform.rotation = Math::EulerAnglesFromQuat(PhysXUtils::FromPhysXQuat(pose.q));
		}
		EntityManager* manager;
		EntityHandle entity;
		physx::PxRigidActor* physxActor;
	};

	class PhysicsScene
	{
	public:
		PhysicsScene()
		{
			physx::PxSceneDesc sceneDesc(GPhysXSDK->getTolerancesScale());
			sceneDesc.flags |= (
				physx::PxSceneFlag::eENABLE_CCD |
				physx::PxSceneFlag::eENABLE_PCM |
				physx::PxSceneFlag::eENABLE_ENHANCED_DETERMINISM |
				physx::PxSceneFlag::eENABLE_ACTIVE_ACTORS);
			sceneDesc.gravity = physx::PxVec3(0.0f, 0.0f, -9.81f);
			sceneDesc.cpuDispatcher = GPhysXCpuDispatcher;
			sceneDesc.filterShader = physx::PxDefaultSimulationFilterShader;
			physxScene = GPhysXSDK->createScene(sceneDesc);
			ASSERT(sceneDesc.isValid());

			material = GPhysXSDK->createMaterial(0.5f, 0.5f, 0.6f);
			physx::PxRigidStatic* groundPlane = PxCreatePlane(*GPhysXSDK, physx::PxPlane(0, 0, 1, 0), *material);
			physxScene->addActor(*groundPlane);
		}
		~PhysicsScene()
		{
			Clear();
			PX_RELEASE(physxScene);
		}
		void CreateActor(EntityManager* manager, EntityHandle entity)
		{
			const TransformComponent& transform = manager->GetComponent<TransformComponent>(entity);
			const RigidBodyComponent& rigidBodyComponent = manager->GetComponent<RigidBodyComponent>(entity);
			const BoxColliderComponent& boxColliderComponent = manager->GetComponent<BoxColliderComponent>(entity);

			physx::PxRigidActor* physxActor = nullptr;
			if (rigidBodyComponent.type == RigidBodyComponent::Type::Static)
			{
				physx::PxRigidStatic* rigidStatic = GPhysXSDK->createRigidStatic(PhysXUtils::ToPhysXTransform(transform.world));
				physxActor = rigidStatic;
			}
			else
			{
				physx::PxTransform localTm(PhysXUtils::ToPhysXVec3(transform.position));
				physx::PxRigidDynamic* rigidDynamic = GPhysXSDK->createRigidDynamic(localTm);
				rigidDynamic->setLinearDamping(rigidBodyComponent.linearDamping);
				rigidDynamic->setAngularDamping(rigidBodyComponent.angularDamping);
				rigidDynamic->setRigidBodyFlag(physx::PxRigidBodyFlag::eKINEMATIC, rigidBodyComponent.isKinematic);
				rigidDynamic->setActorFlag(physx::PxActorFlag::eDISABLE_GRAVITY, rigidBodyComponent.disableGravity);
				physx::PxRigidBodyExt::updateMassAndInertia(*rigidDynamic, 10.0f);
				physxActor = rigidDynamic;
			}
			physx::PxShape* shape = GPhysXSDK->createShape(physx::PxBoxGeometry(boxColliderComponent.halfExtent.x, boxColliderComponent.halfExtent.y, boxColliderComponent.halfExtent.z), *material);
			physxActor->attachShape(*shape);

			PhysicsActor* actor = new PhysicsActor(); 
			actor->manager = manager;
			actor->entity = entity;
			actor->physxActor = physxActor;

			physxActor->userData = actor;

			physxScene->addActor(*physxActor);

			shape->release();
		}
		void RemoveActor(EntityHandle entity)
		{
		}
		void Simulate(float deltaTime)
		{
			physxScene->simulate(deltaTime);
			physxScene->fetchResults(true);

			{
				uint32 numActiveActors;
				physx::PxActor** activeActors = physxScene->getActiveActors(numActiveActors);
				for (uint32 i = 0; i < numActiveActors; i++)
				{
					PhysicsActor* actor = (PhysicsActor*)activeActors[i]->userData;
					if (actor)
					{
						actor->SynchronizeTransform();
					}
				}
			}

			physx::PxSimulationStatistics simulationStats;
			physxScene->getSimulationStatistics(simulationStats);
		}
		void Clear()
		{

		}
	private:
		physx::PxScene* physxScene;
		physx::PxMaterial* material;
	};
}