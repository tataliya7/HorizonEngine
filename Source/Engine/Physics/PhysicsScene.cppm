module;

#include <PxPhysicsAPI.h>

#include "Core/CoreDefinitions.h"

export module HorizonEngine.Physics.Scene;

import HorizonEngine.Core;
import HorizonEngine.Entity;
import HorizonEngine.Physics.Core;
import HorizonEngine.Physics.PhysXUtils;

export namespace HE
{
	class PhysicsActor
	{
	public:
		void SynchronizeTransform()
		{
			TransformComponent& transform = manager->GetComponent<TransformComponent>(entity);
			manager->AddOrReplaceComponent<TransformDirtyComponent>(entity);

			physx::PxTransform pose = physxActor->getGlobalPose();

			transform.position = PhysXUtils::FromPhysXVector(pose.p);
			transform.rotation = Math::EulerAnglesFromQuaternion(PhysXUtils::FromPhysXQuat(pose.q));
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
			const RigidBodyComponent& rigidBody = manager->GetComponent<RigidBodyComponent>(entity);

			physx::PxRigidActor* physxActor = nullptr;
			if (rigidBody.type == RigidBodyComponent::Type::Static)
			{
				physx::PxRigidStatic* rigidStatic = GPhysXSDK->createRigidStatic(PhysXUtils::ToPhysXTransform(transform.world));
				physxActor = rigidStatic;
			}
			else
			{
				physx::PxTransform localTm(PhysXUtils::ToPhysXVec3(transform.position));
				physx::PxRigidDynamic* rigidDynamic = GPhysXSDK->createRigidDynamic(localTm);
				rigidDynamic->setLinearDamping(rigidBody.linearDamping);
				rigidDynamic->setAngularDamping(rigidBody.angularDamping);
				rigidDynamic->setRigidBodyFlag(physx::PxRigidBodyFlag::eKINEMATIC, rigidBody.isKinematic);
				rigidDynamic->setActorFlag(physx::PxActorFlag::eDISABLE_GRAVITY, rigidBody.disableGravity);
				physx::PxRigidBodyExt::updateMassAndInertia(*rigidDynamic, 10.0f);
				physxActor = rigidDynamic;
			}
			physx::PxShape* shape = nullptr;
			
			switch (rigidBody.shape)
			{
			case RigidBodyComponent::CollisionShape::Box:
			{
				physx::PxBoxGeometry geometry = physx::PxBoxGeometry(
					rigidBody.boxCollider.halfExtent.x,
					rigidBody.boxCollider.halfExtent.y,
					rigidBody.boxCollider.halfExtent.z);
				shape = GPhysXSDK->createShape(geometry, *material);
				break;
			}
			case RigidBodyComponent::CollisionShape::Sphere:
			{
				physx::PxSphereGeometry geometry = physx::PxSphereGeometry(rigidBody.sphereCollider.radius);
				shape = GPhysXSDK->createShape(geometry, *material);
				break;
			}
			case RigidBodyComponent::CollisionShape::Capsule:
			{
				physx::PxCapsuleGeometry geometry = physx::PxCapsuleGeometry(rigidBody.capsuleCollider.radius, (rigidBody.capsuleCollider.height / 2.0f));
				shape = GPhysXSDK->createShape(geometry, *material);
			}
				break;
			case RigidBodyComponent::CollisionShape::Mesh:
			{
				break;
			}
			}
			
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