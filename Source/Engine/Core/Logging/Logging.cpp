module;

#include "CoreCommon.h"

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <filesystem>

module HorizonEngine.Core.Logging;

namespace HE
{
	bool LogSystemInit()
	{
		std::string logsDirectory = "Logs";
		if (!std::filesystem::exists(logsDirectory))
		{
			std::filesystem::create_directories(logsDirectory);
		}

		std::vector<spdlog::sink_ptr> sinks =
		{
			std::make_shared<spdlog::sinks::stdout_color_sink_mt>(),
			std::make_shared<spdlog::sinks::basic_file_sink_mt>("Logs/Horizon.log", true),
		};

		sinks[0]->set_pattern("%^[%T] %n: %v%$");
		sinks[1]->set_pattern("[%T] [%l] %n: %v");

		auto colorSink = static_cast<spdlog::sinks::stdout_color_sink_mt*>(sinks[0].get());
		colorSink->set_color(spdlog::level::info, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
		colorSink->set_color(spdlog::level::warn, FOREGROUND_RED | FOREGROUND_GREEN);
		colorSink->set_color(spdlog::level::err, FOREGROUND_RED);

		gLogger = std::make_shared<spdlog::logger>("Console Logger", begin(sinks), end(sinks));
		gLogger->set_level(spdlog::level::trace);

		spdlog::register_logger(gLogger);

		return true;
	}

	void LogSystemExit()
	{
		gLogger = nullptr;
	}
}
