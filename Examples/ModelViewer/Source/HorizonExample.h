#pragma once

#include <HorizonExampleBase.h>

#define HORIZON_EXAMPLE_NAME "ModelViewer"

namespace HE
{
	class HorizonExample : public HorizonExampleBase
	{
	public:

		HorizonExample();
		~HorizonExample();

		void Setup() override;
		void Clear() override;

		void OnUpdate(float deltaTime) override;
		void OnRender() override;
		void OnDrawUI() override;

	private:

		void DrawEntityNodeUI(EntityHandle entity);

		Scene* scene;

		SceneView* view;

		EntityHandle mainCamera;

		EntityHandle selectedEntity;
	};
}
