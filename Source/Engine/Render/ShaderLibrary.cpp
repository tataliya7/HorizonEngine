module;

#include "Core/CoreDefinitions.h"

module HorizonEngine.Render.ShaderSystem;

#define CREATE_SHADER(shader, filename, entry, stage) { RenderBackendShaderDesc shaderDesc = {};LoadShaderSourceFromFile(filename, source); CompileShader(shaderCompiler,source,HE_TEXT(entry),stage,ShaderRepresentation::SPIRV,includeDirs,defines,&shaderDesc.stages[(uint32)stage]); shaderDesc.entryPoints[(uint32)stage] = entry; shader = RenderBackendCreateShader(renderBackend, deviceMask, &shaderDesc, "shader");}

namespace HE
{
	ShaderLibrary::ShaderLibrary(RenderBackend* backend, ShaderCompiler* compiler)
		: renderBackend(backend)
		, shaderCompiler(compiler) 
	{
		std::vector<uint8> source;
		std::vector<const wchar*> includeDirs;
		std::vector<const wchar*> defines;
		includeDirs.push_back(HE_TEXT("../../../Shaders"));

		uint32 deviceMask = ~0u;
	
		RenderBackendShaderHandle EquirectangularToCubemapCS;
		RenderBackendShaderHandle DownsampleCubemapCS;
		RenderBackendShaderHandle ComputeEnviromentIrradianceCS;
		RenderBackendShaderHandle FilterEnviromentMapCS;

		CREATE_SHADER(EquirectangularToCubemapCS, "../../../Shaders/EquirectangularToCubemap.hsf", "EquirectangularToCubemapCS", RenderBackendShaderStage::Compute);
		CREATE_SHADER(DownsampleCubemapCS, "../../../Shaders/DownsampleCubemap.hsf", "DownsampleCubemapCS", RenderBackendShaderStage::Compute);
		CREATE_SHADER(ComputeEnviromentIrradianceCS, "../../../Shaders/ComputeEnviromentIrradiance.hsf", "ComputeEnviromentIrradianceCS", RenderBackendShaderStage::Compute);
		CREATE_SHADER(FilterEnviromentMapCS, "../../../Shaders/FilterEnviromentMap.hsf", "FilterEnviromentMapCS", RenderBackendShaderStage::Compute);
		
		loadedShaders.emplace("EquirectangularToCubemapCS", EquirectangularToCubemapCS);
		loadedShaders.emplace("DownsampleCubemapCS", DownsampleCubemapCS);
		loadedShaders.emplace("ComputeEnviromentIrradianceCS", ComputeEnviromentIrradianceCS);
		loadedShaders.emplace("FilterEnviromentMapCS", FilterEnviromentMapCS);
	}

	bool ShaderLibrary::LoadShader(const char* filename, const wchar* entry)
	{
		/*ShaderCreateInfo shaderInfo = {};
		
		std::vector<const wchar*> includeDirs;
		std::vector<const wchar*> defines;

		includeDirs.push_back(HE_TEXT("../../../Shaders"));
		includeDirs.push_back(HE_TEXT("../../../Shaders/HybridRenderPipeline"));

		RenderBackendShaderDesc brdfLutShaderDesc;
		LoadShaderSourceFromFile(filename, shaderInfo.code);

		CompileShader(
			shaderCompiler,
			shaderInfo.code,
			entry,
			RenderBackendShaderStage::Compute,
			ShaderRepresentation::SPIRV,
			includeDirs,
			defines,
			&brdfLutShaderDesc.stages[(uint32)RenderBackendShaderStage::Compute]);
		brdfLutShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Compute] = "BRDFLutCS";
		brdfLutShader = RenderBackendCreateShader(renderBackend, deviceMask, &brdfLutShaderDesc, "BRDFLutShader");

		Shader* shader = new Shader(shaderInfo);
		ASSERT(loadedShaders.find(name) == loadedShaders.end());
		loadedShaders[name] = shader;*/
		return true;
	}
}