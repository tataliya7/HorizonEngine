//#include "Plugin.h"
//
//namespace HE
//{
//namespace PluginSystem
//{
//std::vector<Plugin> gPlugins;
//
//static bool LoadPluginFromFile(const char* path, bool hotReloadable, bool hotReload, Plugin* outPlugin)
//{
//    *outPlugin = {};
//
//    const auto dll = OpenDLL(path);
//    if (dll)
//    {
//        return false;
//    }
//
//    const auto loadCallback = GetSymbolFromDLL(dll, "Load");
//    const auto unloadCallback = GetSymbolFromDLL(dll, "Unload");
//    if (!loadCallback || !unloadCallback)
//    {
//        HE_LOG_ERROR("Could not load plugin '{}'. LoadCallback / UnloadCallback() not found.", path);
//        CloseDLL(dll);
//        return false;
//    }
//
//    outPlugin->name = GetBaseName(path);
//    outPlugin->path = path;
//    outPlugin->dll = dll;
//    outPlugin->loadCallback = (PluginCallback)loadCallback;
//    outPlugin->unloadCallback = (PluginCallback)unloadCallback;
//    outPlugin->hotReloadable = hotReloadable;
//
//    return true;
//}
//
//void PluginSystem::LoadAllPlugins(const char* directory)
//{
//    const char* pluginExtension = "*.dll";
//    std::vector<std::string> pluginPaths;
//    FindFiles(directory, pluginExtension, pluginPaths);
//    for (const auto& path : pluginPaths)
//    {
//        LoadPlugin(path.c_str(), true);
//    }
//}
//
//PluginHandle LoadPlugin(const char* path, bool hotReloadable)
//{
//    if (hotReloadable)
//    {
//        // Cleanup
//    }
//
//    Plugin plugin;
//    if (!LoadPluginFromFile(path, hotReloadable, false, &plugin))
//    {
//        return 0;
//    }
//
//    plugin.loadCallback();
//
//    uint32 index = (uint32)gPlugins.size();
//    gPlugins.push_back(plugin);
//
//    plugin.handle = index + 1;
//
//    return plugin.handle;
//}
//
//void ReloadPlugin(PluginHandle handle)
//{
//
//}
//
//void UnloadPlugin(PluginHandle handle)
//{
//    uint32 index = handle - 1;
//    Plugin& plugin = gPlugins[index];
//
//    if (plugin.unloadCallback)
//    {
//        plugin.unloadCallback();
//    }
//
//    CloseDLL(plugin.dll);
//    plugin = {};
//}
//
//bool HotReload()
//{
//    bool flag = false;
//    for (const auto& plugin : gPlugins)
//    {
//        if (!plugin.hotReloadable)
//        {
//            continue;
//        }
//        FileTime lastModifiedTime = GetFileAttributeData(plugin.path.c_str()).lastModifiedTime;
//        if (lastModifiedTime.time == plugin.lastModifiedTime.time)
//        {
//            continue;
//        }
//        ReloadPlugin(plugin.handle);
//        flag = true;
//    }
//    return flag;
//}
//}
//}