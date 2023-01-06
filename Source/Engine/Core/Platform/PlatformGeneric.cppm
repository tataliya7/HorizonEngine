module;

#include <string>
#include <vector>

export module HorizonEngine.Core.Platform.Generic;

import HorizonEngine.Core.Types;

export namespace HE
{
	struct DateTime
	{
		uint64 time;
	};

	struct FileStatData
	{
		bool isValid;
		bool isDirectory;
		bool isReadOnly;
		DateTime creationTime;
		DateTime accessTime;
		DateTime modificationTime;
		int64 size;
	};

	const char* GetBaseName(const char* path);
	std::string GetDirectory(const std::string& path);
	uint32 GetNumberOfProcessors();
	FileStatData GetFileAttributeData(const char* path);
	void SuspendCurrentThread(float seconds);
	void YieldCPU();
	uint32 GetCurrentThreadID();
	void SwitchToAnotherFiber(uint64 handle);
	uint64 OpenDLL(const char* path);
	void* GetSymbolFromDLL(uint64 handle, const char* name);
	void CloseDLL(uint64 handle);
	bool GetExePath(char* path, uint32 size);
	void FindFiles(const std::string& directory, const std::string& extension, std::vector<std::string>& outPaths);
}