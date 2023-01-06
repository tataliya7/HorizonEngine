#pragma once

#include "EditorWindow.h"
#include "Events/Event.h"
#include "Events/MouseEvent.h"
#include "Core/Timestep.h"
#include "Math/HorizonMath.h"

namespace Horizon
{
	class SceneWindow : public EditorWindow
	{
	public:
		SceneWindow(const String& title, Editor* editor);
		~SceneWindow() = default;
		void Draw() override;
		void OnEvent(Event& e);
		bool OnMouseScroll(MouseScrollEvent& e);
		bool OnMouseMove(MouseMoveEvent& e);
		bool OnMouseButtonPress(MouseButtonPressEvent& e);
		bool IsInViewport(const Vector2& pos);
		bool IsFocused() { return mIsFocused; }
		bool Pick(MouseButtonPressEvent& e);
		void SetGizmoOperationType(int type) { mGizmoOperationType = type; }
		const Vector2& GetViewportSize() const { return mViewportSize; }
	private:
		float GetSnapValue();
		bool mIsFocused = true;
		int mGizmoOperationType;
		Vector2 mViewportPos;
		Vector2 mViewportSize;
	};
}
