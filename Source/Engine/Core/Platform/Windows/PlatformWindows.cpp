module;

#if HE_PLATFORM_WINDOWS

#define NOMINMAX
#include <windows.h>
#include <string>

module HorizonEngine.Core.Platform.Generic;

import HorizonEngine.Core.Logging;

namespace HE
{
    uint64 gMainThreadID = 0;

    const char* GetBaseName(const char* path)
    {
        const char* fslash = strrchr(path, '/');
        const char* bslash = strrchr(path, '\\');
        const char* slash = fslash > bslash ? fslash : bslash;
        return slash ? slash + 1 : path;
    }

    std::wstring GetDirectory(const std::wstring& path)
    {
        const auto& index = std::max(path.rfind('\\'), path.rfind('/'));
        if (std::wstring::npos != index)
        {
            return path.substr(0, index);
        }
        return TEXT("");
    }

    FileStatData GetFileAttributeData(const char* path)
    {
        FileStatData data = {};

        wchar_t filename[100];
        wsprintf(filename, L"%s", path);

        WIN32_FILE_ATTRIBUTE_DATA win32FileAttributeData;
        DWORD result = GetFileAttributesExW(filename, GetFileExInfoStandard, &win32FileAttributeData);

        data.isValid = (result != 0);
        data.isDirectory = (win32FileAttributeData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
        if (data.isValid)
        {
            data.modificationTime.time = ((uint64)(win32FileAttributeData.ftLastWriteTime.dwHighDateTime) << 32) | win32FileAttributeData.ftLastWriteTime.dwLowDateTime;
        }
        if (data.isValid && !data.isDirectory)
        {
            data.size = ((uint64)(win32FileAttributeData.nFileSizeHigh) << 32) | win32FileAttributeData.nFileSizeLow;
        }

        return data;
    }

    bool IsInMainThread()
    {
        return GetCurrentThreadId() == (DWORD)gMainThreadID;
    }
   
    uint32 GetNumberOfProcessors()
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return si.dwNumberOfProcessors;
    }

    void SuspendCurrentThread(float seconds)
    {
        Sleep((DWORD)(seconds * 1000.0f + 0.5f));
    }

    void YieldCPU()
    {
        YieldProcessor();
    }

    uint32 GetCurrentThreadID()
    {
        return GetCurrentThreadId();
    }

    void SwitchToAnotherFiber(uint64 handle)
    {
        SwitchToFiber((void*)handle);
    }

    uint64 OpenDLL(const char* path)
    {
        const size_t size = strlen(path) + 1;
        wchar_t* pathW = new wchar_t[size];
        mbstowcs(pathW, path, size);
        DWORD len = GetFullPathNameW(pathW, 0, 0, 0);
        wchar_t* fullPathW = new wchar_t[len];
        GetFullPathNameW(pathW, len, fullPathW, 0);
        HMODULE handle = LoadLibraryExW(fullPathW, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!handle)
        {
            const DWORD err = GetLastError();
            HE_LOG_ERROR("Error occurs when load {}. LoadLibraryEx() returns error code: {}.", path, err);
        }
        delete[] pathW;
        delete[] fullPathW;
        return (uint64)handle;
    }

    void* GetSymbolFromDLL(uint64 handle, const char* name)
    {
        return GetProcAddress((HMODULE)handle, name);
    }

    void CloseDLL(uint64 handle)
    {
        FreeLibrary((HMODULE)handle);
    }

    bool GetExePath(char* path, uint32 size)
    {
        const DWORD result = GetModuleFileNameA(0, path, size);
        if (result == 0) 
        {
            return false;
        }
        else if (result >= size)
        {
            return false;
        }
        return true;
    }

    void FindFiles(const std::wstring& directory, const std::wstring& extension, std::vector<std::wstring>& outPaths)
    {
        WIN32_FIND_DATA fileData;
        HANDLE hFind;
        std::wstring filename = directory + extension;
        hFind = FindFirstFile(filename.data(), &fileData);
        if (hFind != INVALID_HANDLE_VALUE)
        {
            outPaths.push_back(directory + fileData.cFileName);
            while (FindNextFile(hFind, &fileData) != 0)
            {
                outPaths.push_back(directory + fileData.cFileName);
            }
        }
    }
}
#endif