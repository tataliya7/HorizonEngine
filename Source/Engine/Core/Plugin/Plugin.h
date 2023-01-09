#pragma once

//#include "Core/CoreCommon.h"
//
//namespace HE
//{
//using PluginHandle = uint32;
//
//namespace PluginSystem
//{
//    using PluginCallback = void(*)(void);
//    struct Plugin
//    {
//        PluginHandle handle;
//        bool hotReloadable;
//        std::string path;
//        std::string name;
//        uint64 dll;
//        DateTime modificationTime;
//        PluginCallback loadCallback;
//        PluginCallback unloadCallback;
//    };
//
//    void LoadAllPlugins(const char* directory);
//    PluginHandle LoadPlugin(const char* path, bool hotReloadable);
//    void ReloadPlugin(PluginHandle handle);
//    void UnloadPlugin(PluginHandle handle);
//    bool HotReload();
//}
//}