#pragma once

#include "ECS/ECSCommon.h"
#include "ECS/Entity.h"

#include <functional>

namespace HE
{
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
