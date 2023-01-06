#pragma once

#ifndef HE_LOG
#define HE_LOG(level, ...)
#endif

#ifndef HE_LOG_VERBOSE
#define HE_LOG_VERBOSE(...)    ::HE::LogVerbose(__VA_ARGS__);
#endif

#ifndef HE_LOG_INFO
#define HE_LOG_INFO(...)       ::HE::LogInfo(__VA_ARGS__);
#endif

#ifndef HE_LOG_WARNING
#define HE_LOG_WARNING(...)    ::HE::LogWarning(__VA_ARGS__);
#endif

#ifndef HE_LOG_ERROR
#define HE_LOG_ERROR(...)      ::HE::LogError(__VA_ARGS__);
#endif

#ifndef HE_LOG_FATAL
#define HE_LOG_FATAL(...)      ::HE::LogFatal(__VA_ARGS__);
#endif