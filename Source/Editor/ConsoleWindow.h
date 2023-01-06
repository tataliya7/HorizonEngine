#pragma once

#include "Editor.h"

namespace Horizon
{
	class ConsoleWindow : public EditorWindow
	{
	public:
		static const uint32 kMaxMessageCount = 100;
		class Message
		{
		public:
			enum class Level : uint32
			{
				Debug,
				Info,
				Warning,
				Error,
				Fatal,
			};
			Message(const String& msg, Level level) : mContent(msg), mLevel(level) {}
		private:
			friend class ConsoleWindow;
			void Draw() const;
			String mContent;
			Level mLevel;
		};
		ConsoleWindow(const String& title, Editor* editor);
		~ConsoleWindow() = default;
		void Draw() override;
		void Flush();
		void AddMessage(const Message& msg);
	private:
		bool mScrollToBottom = true;
		std::vector<Message> mMessages;
	};
}