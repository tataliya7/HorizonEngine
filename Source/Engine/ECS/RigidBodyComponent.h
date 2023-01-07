#pragma once

#include "ECS/ECSCommon.h"
#include "ECS/Entity.h"

namespace HE
{
	struct BoxColliderComponent
	{
		Vector3 halfExtent = { 0.5f, 0.5f, 0.5f };
		Vector3 offset = { 0.0f, 0.0f, 0.0f };
		
		BoxColliderComponent() = default;
		BoxColliderComponent(const BoxColliderComponent& other) = default;
	};

	struct SphereColliderComponent
	{
		SphereColliderComponent() = default;
		SphereColliderComponent(const SphereColliderComponent& other) = default;
	};

	struct CapsuleColliderComponent
	{
		CapsuleColliderComponent() = default;
		CapsuleColliderComponent(const CapsuleColliderComponent& other) = default;
	};

	struct MeshColliderComponent
	{
		MeshColliderComponent() = default;
		MeshColliderComponent(const MeshColliderComponent& other) = default;
	};

	struct RigidBodyComponent
	{
		enum class Type 
		{ 
			Static  = 0,
			Dynamic = 1,
		};

		Type type = Type::Static;

		bool disableGravity = false;
		bool isKinematic = false;

		float mass = 1.0f;
		float linearDamping = 0.01f;
		float angularDamping = 0.05f;

		RigidBodyComponent() = default;
		RigidBodyComponent(const RigidBodyComponent& other) = default;
	};
}
