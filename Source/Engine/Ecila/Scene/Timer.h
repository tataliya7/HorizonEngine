#pragma once

#include <chrono>

namespace Ecila
{
	class Timer
	{
	public:

		Timer()
		{
			Reset();
		}

		void Timer::Reset()
		{
			mStartTime = clock::now();
		}

		float Timer::ElapsedSeconds()
		{
			return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - mStartTime).count() * 0.001f * 0.001f * 0.001f;
		}

		float Timer::ElapsedMilliseconds()
		{
			return ElapsedSeconds() * 1000.0f;
		}

	private:

		using clock = std::chrono::steady_clock;

		clock::time_point mStartTime;
	};
}
