module;

#define SPDLOG_USE_STD_FORMAT
#include <spdlog/spdlog.h>

export module HorizonEngine.Core.Logging;

std::shared_ptr<spdlog::logger> gLogger = nullptr;

export namespace HE
{
    enum class LogLevel
    {
        Verbose,
        Info,
        Warning,
        Error,
        Fatal,
    };

    bool LogSystemInit();

    void LogSystemExit();

    template<typename... Args>
    void LogVerbose(Args&&... args)
    {
        gLogger->debug(std::forward<Args>(args)...);
    }

    template<typename... Args>
    void LogInfo(Args&&... args)
    {
        gLogger->info(std::forward<Args>(args)...);
    }

    template<typename... Args>
    void LogWarning(Args&&... args)
    {
        gLogger->warn(std::forward<Args>(args)...);
    }

    template<typename... Args>
    void LogError(Args&&... args)
    {
        gLogger->error(std::forward<Args>(args)...);
    }

    template<typename... Args>
    void LogFatal(Args&&... args)
    {
        gLogger->critical(std::forward<Args>(args)...);
    }
}