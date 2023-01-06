#pragma once

#include "EditorWindow.h"

namespace Horizon
{
	class InspectorWindow : public EditorWindow
	{
	public:
		InspectorWindow(const String& title, Editor* editor);
		~InspectorWindow() = default;
		void Draw() override;
	private:
		void DrawComponents_Internal(Entity* entity);
	};
}