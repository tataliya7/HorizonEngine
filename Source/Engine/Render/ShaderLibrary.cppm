module;

#include <unordered_map>

export module HorizonEngine.Render.ShaderSystem;

import HorizonEngine.Core;
import HorizonEngine.Render.Core;

export namespace HE
{
	struct ShaderCreateInfo
	{
		std::vector<uint8> code;
		uint32 codeSize;
	};

	/** A compiled shader. */
	class Shader
	{
	public:
		Shader(const ShaderCreateInfo& info);
		~Shader();
		uint64 GetHash() const;
	private:
		RenderBackendShaderHandle handle;
	};

	struct ShaderMacro
	{
		std::string name;
		std::string value;
	};

	typedef std::vector<ShaderMacro> ShaderMacros;

	class ShaderLibrary
	{
	public:
		ShaderLibrary(RenderBackend* backend, ShaderCompiler* compiler);
		~ShaderLibrary() {}
		bool LoadShader(const char* filename, const wchar* entry);
		//bool ReloadShader();
		//void UnloadShader();
		//void Clear();
		/*Shader* GetShader(const std::string& name) const
		{
			ASSERT(loadedShaders.find(name) != loadedShaders.end());
			return loadedShaders[name];
		}*/
		RenderBackendShaderHandle GetShader(const std::string& name)
		{
			ASSERT(loadedShaders.find(name) != loadedShaders.end());
			return loadedShaders[name];
		}
	private:
		RenderBackend* renderBackend;
		ShaderCompiler* shaderCompiler;
		//std::unordered_map<std::string, Shader*> loadedShaders;
		std::unordered_map<std::string, RenderBackendShaderHandle> loadedShaders;
	};

	ShaderLibrary* GGlobalShaderLibrary = nullptr;

	ShaderLibrary* GetGlobalShaderLibrary()
	{
		ASSERT(GGlobalShaderLibrary);
		return GGlobalShaderLibrary;
	}
}