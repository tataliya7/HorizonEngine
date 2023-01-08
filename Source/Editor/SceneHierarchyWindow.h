#pragma once

namespace HE
{
	class SceneHierarchyWindow
	{
	public:

		SceneHierarchyWindow(const String& title, Editor* editor);

		~SceneHierarchyWindow();

		void OnDrawUI();

	private:

		void DrawEntityNode(Entity* entity, EntityManager* entityManager);
		void DrawEntityCreateMenu();

		bool open;
		bool focused;
	};
}
