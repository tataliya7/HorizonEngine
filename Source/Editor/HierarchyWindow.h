#pragma once

#include "EditorWindow.h"

namespace Horizon
{
	class SceneHierarchyWindow : public EditorWindow
	{
	public:

		SceneHierarchyWindow(const String& title, Editor* editor);

		~SceneHierarchyWindow();

		void Draw() override;

	private:

		void DrawEntityNode_Internal(Entity* entity, EntityManager* entityManager);
	};
}
