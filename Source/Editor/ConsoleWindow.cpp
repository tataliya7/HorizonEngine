#include "ConsoleWindow.h"

namespace Horizon
{
	ConsoleWindow::ConsoleWindow(const String& title, Editor* editor)
		: EditorWindow(title, editor)
	{
	}

	void ConsoleWindow::AddMessage(const Message& message)
	{
		mMessages.push_back(message);
		if ((uint32)mMessages.size() > kMaxMessageCount)
		{
			mMessages.erase(mMessages.begin());
		}
	}

	void ConsoleWindow::Flush()
	{
		mMessages.clear();
	}

	void ConsoleWindow::Draw()
	{
		ImGui::Begin(mTitle.c_str(), &mActive);
		{
			ImGui::Separator();

			ImGui::BeginChild("ScrollRegion", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
			{
				for (uint64 i = 0; i < mMessages.size(); i++)
				{
					mMessages[i].Draw();
				}
				ImGui::Text("[SceneManager] load glft: DamagedHelmet.glft");
				if (mScrollToBottom && ImGui::GetScrollMaxY() > 0)
				{
					ImGui::SetScrollY(ImGui::GetScrollMaxY());
					// mScrollToBottom = false;
				}
			}
			ImGui::EndChild();

		}
		ImGui::End();
	}

	void ConsoleWindow::Message::Draw() const 
	{
		ImGui::PushID(this);
		ImGui::Text("%s", mContent.c_str());
	}
}