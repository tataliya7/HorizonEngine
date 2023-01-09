module;

#include "Core/CoreDefinitions.h"

export module HorizonEngine.Entity.Components;

import <string>;
import HorizonEngine.Core;
import HorizonEngine.Entity.Manager;

export namespace HE
{
	struct NameComponent
	{
		std::string name;
		NameComponent() = default;
		NameComponent(const NameComponent& other) = default;
		NameComponent(const std::string& name) : name(name) {}
		inline operator std::string& () { return name; }
		inline operator const std::string& () const { return name; }
		inline void operator=(const std::string& str) { name = str; }
		inline bool operator==(const std::string& str) const { return name.compare(str) == 0; }
	};

	struct TransformComponent
	{
		Vector3 position;
		Vector3 rotation;
		Vector3 scale;
		Matrix4x4 world;

		TransformComponent()
			: position(0.0, 0.0, 0.0)
			, rotation(0.0, 0.0, 0.0)
			, scale(1.0, 1.0, 1.0)
		{
			using namespace entt;
			auto factory = entt::meta<TransformComponent>();
			factory.data<&TransformComponent::scale, entt::as_ref_t>("Scale"_hs)
				.prop("Name"_hs, std::string("Scale"));
			factory.data<&TransformComponent::rotation, entt::as_ref_t>("Rotation"_hs)
				.prop("Name"_hs, std::string("Rotation"));
			factory.data<&TransformComponent::position, entt::as_ref_t>("Position"_hs)
				.prop("Name"_hs, std::string("Position"));
		}

		void Update()
		{
			world = Math::Compose(position, Quaternion(Math::DegreesToRadians(rotation)), scale);
		}
	};

	struct TransformDirtyComponent
	{

	};

	struct SceneHierarchyComponent
	{
		uint32 depth;
		uint32 numChildren;
		EntityHandle parent;
		EntityHandle first;
		EntityHandle next;
		EntityHandle prev;

		SceneHierarchyComponent()
			: depth(0)
			, numChildren(0)
			, parent()
			, first()
			, next()
			, prev()
		{
			using namespace entt;
			auto factory = entt::meta<SceneHierarchyComponent>();
			factory.data<&SceneHierarchyComponent::prev, entt::as_ref_t>("Prev"_hs)
				.prop("Name"_hs, std::string("Prev"));
			factory.data<&SceneHierarchyComponent::next, entt::as_ref_t>("Next"_hs)
				.prop("Name"_hs, std::string("Next"));
			factory.data<&SceneHierarchyComponent::first, entt::as_ref_t>("First"_hs)
				.prop("Name"_hs, std::string("First"));
			factory.data<&SceneHierarchyComponent::parent, entt::as_ref_t>("Parent"_hs)
				.prop("Name"_hs, std::string("Parent"));
			factory.data<&SceneHierarchyComponent::numChildren, entt::as_ref_t>("Num Children"_hs)
				.prop("Name"_hs, std::string("Num Children"));
			factory.data<&SceneHierarchyComponent::depth, entt::as_ref_t>("Depth"_hs)
				.prop("Name"_hs, std::string("Depth"));
		}
	};
	
	struct CameraComponent
	{
		enum class Type
		{
			Perpective,
			Orthographic,
		};
		Type type;
		float nearPlane;
		float farPlane;
		float fieldOfView;
		float aspectRatio;
		bool overrideAspectRatio;

		CameraComponent()
		{
			using namespace entt;
			auto factory = entt::meta<CameraComponent>();
			factory.data<&CameraComponent::overrideAspectRatio, entt::as_ref_t>("Override Aspect Ratio"_hs)
				.prop("Name"_hs, std::string("Override Aspect Ratio"));
			factory.data<&CameraComponent::aspectRatio, entt::as_ref_t>("Aspect Ratio"_hs)
				.prop("Name"_hs, std::string("Aspect Ratio"));
			factory.data<&CameraComponent::fieldOfView, entt::as_ref_t>("Field of View"_hs)
				.prop("Name"_hs, std::string("Field of View"));
			factory.data<&CameraComponent::farPlane, entt::as_ref_t>("Far Plane"_hs)
				.prop("Name"_hs, std::string("Far Plane"));
			factory.data<&CameraComponent::nearPlane, entt::as_ref_t>("Near Plane"_hs)
				.prop("Name"_hs, std::string("Near Plane"));
		}
	};

	class LightRenderProxy;

	struct DirectionalLightComponent
	{
		Vector3 color;
		float intensity;
		DirectionalLightComponent()
		{
			using namespace entt;
			auto factory = entt::meta<DirectionalLightComponent>();
			factory.data<&DirectionalLightComponent::intensity, entt::as_ref_t>("Intensity"_hs)
				.prop("Name"_hs, std::string("Intensity"));
			factory.data<&DirectionalLightComponent::color, entt::as_ref_t>("Color"_hs)
				.prop("Name"_hs, std::string("Color"));
		}
		bool rayTracingShadows = false;
		uint32 numDynamicShadowCascades = 4;
		uint32 shadowMapSize     = 4096;
		float cascadeSplitLambda = 0.95f;

		LightRenderProxy* proxy;
	};

	class SkyLightRenderProxy;

	struct SkyLightComponent
	{
		std::string cubemap;
		uint32 cubemapResolution;

		SkyLightComponent()
		{
			using namespace entt;
			auto factory = entt::meta<SkyLightComponent>();
			factory.data<&SkyLightComponent::cubemapResolution, entt::as_ref_t>("Cubemap Resolution"_hs)
				.prop("Name"_hs, std::string("Cubemap Resolution"));
			factory.data<&SkyLightComponent::cubemap, entt::as_ref_t>("Cubemap"_hs)
				.prop("Name"_hs, std::string("Cubemap"));
		}

		void SetCubemap(std::string newCubemap)
		{
			if (cubemap != newCubemap)
			{
				cubemap = newCubemap;
				SetDirty();
			}
		}

		void SetDirty()
		{
			valid = false;
		}

		SkyLightRenderProxy* proxy;

		bool valid;
	};

	class MeshRenderProxy;

	struct StaticMeshComponent
	{
		std::string meshSource;

		MeshRenderProxy* proxy;

		StaticMeshComponent()
		{
			using namespace entt;
			auto factory = entt::meta<StaticMeshComponent>();
			factory.data<&StaticMeshComponent::meshSource, entt::as_ref_t>("Mesh Source"_hs)
				.prop("Name"_hs, std::string("Mesh Source"));
		}
	};

	struct AudioListenerComponent
	{
		uint64 id;
		AudioListenerComponent() = default;
		AudioListenerComponent(const AudioListenerComponent& other) = default;
	};

	struct RigidBodyComponent
	{
		enum class Type
		{
			Static = 0,
			Dynamic = 1,
		};

		enum class CollisionShape
		{
			Box = 0,
			Sphere = 1,
			Capsule = 2,
			Mesh = 3,
		};

		struct BoxCollider
		{
			Vector3 halfExtent = Vector3(0.5f, 0.5f, 0.5f);
			Vector3 offset = Vector3(0.0f, 0.0f, 0.0f);
		}; 
		
		struct SphereCollider
		{
			float radius = 1.0f;
		};

		struct CapsuleCollider
		{
			float radius = 1.0f;
			float height = 1.0f;
		};

		Type type = Type::Static;
		CollisionShape shape = CollisionShape::Mesh;

		BoxCollider boxCollider;
		SphereCollider sphereCollider;
		CapsuleCollider capsuleCollider;

		bool disableGravity = false;
		bool isKinematic = false;

		float mass = 1.0f;
		float linearDamping = 0.01f;
		float angularDamping = 0.05f;

		void SetBoxCollider(const Vector3& halfExtent, const Vector3& offset)
		{
			shape = CollisionShape::Box;
			boxCollider.halfExtent = halfExtent;
			boxCollider.offset = offset;
		}

		void SetSphereCollider(float radius)
		{
			shape = CollisionShape::Sphere;
			sphereCollider.radius = radius;
		}

		void SetCapsuleCollider(float radius, float height)
		{
			shape = CollisionShape::Capsule;
			capsuleCollider.radius = radius;
			capsuleCollider.height = height;
		}

		void SetMeshCollider()
		{
			shape = CollisionShape::Mesh;
		}

		RigidBodyComponent() = default;
		RigidBodyComponent(const RigidBodyComponent& other) = default;
	};

	class Scriptable
	{
	public:
		Scriptable() {}
		virtual ~Scriptable() {}
		template<typename Component>
		Component& GetComponent() const
		{
			return manager->GetComponent<Component>(entity);
		}
		template<typename Component>
		Component* TryGetComponent()
		{
			return manager->TryGetComponent<Component>(entity);
		}
		EntityManager* manager;
		EntityHandle entity;
	};

	struct ScriptComponent
	{
		Scriptable* scriptable = nullptr;

		std::function<void()> ConstructorFunc;
		std::function<void()> DeonstructorFunc;

		std::function<void(Scriptable*)> OnCreateFunc;
		std::function<void(Scriptable*)> OnDestroyFunc;
		std::function<void(Scriptable*, float)> OnUpdateFunc;

		ScriptComponent() = default;

		template<typename T>
		void Bind()
		{
			ConstructorFunc = [&]() { scriptable = new T(); };
			DeonstructorFunc = [&]() { if (scriptable) { delete ((T*)scriptable); scriptable = nullptr; } };

			OnCreateFunc = [&](Scriptable* scriptable) { ((T*)scriptable)->OnCreate(); };
			OnDestroyFunc = [&](Scriptable* scriptable) { ((T*)scriptable)->OnDestroy(); };
			OnUpdateFunc = [&](Scriptable* scriptable, float deltaTime) { ((T*)scriptable)->OnUpdate(deltaTime); };
		}
	};
}
