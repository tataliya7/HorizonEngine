module HorizonEngine.Render.ShaderSystem;

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
		{
			RenderBackendShaderDesc computeShaderDesc;
			LoadShaderSourceFromFile("../../../Shaders/EquirectangularToCubemap.hsf", source);
			CompileShader(
				shaderCompiler,
				source,
				HE_TEXT("EquirectangularToCubemapCS"),
				RenderBackendShaderStage::Compute,
				ShaderRepresentation::SPIRV,
				includeDirs,
				defines,
				&computeShaderDesc.stages[(uint32)RenderBackendShaderStage::Compute]);
			computeShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Compute] = "EquirectangularToCubemapCS";
			RenderBackendShaderHandle computeShader = RenderBackendCreateShader(renderBackend, deviceMask, &computeShaderDesc, "EquirectangularToCubemapCS");
			loadedShaders.emplace("EquirectangularToCubemapCS", computeShader);
		}

		{
			RenderBackendShaderDesc computeShaderDesc;
			LoadShaderSourceFromFile("../../../Shaders/DownsampleCubemap.hsf", source);
			CompileShader(
				shaderCompiler,
				source,
				HE_TEXT("DownsampleCubemapCS"),
				RenderBackendShaderStage::Compute,
				ShaderRepresentation::SPIRV,
				includeDirs,
				defines,
				&computeShaderDesc.stages[(uint32)RenderBackendShaderStage::Compute]);
			computeShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Compute] = "DownsampleCubemapCS";
			RenderBackendShaderHandle computeShader = RenderBackendCreateShader(renderBackend, deviceMask, &computeShaderDesc, "DownsampleCubemapCS");
			loadedShaders.emplace("DownsampleCubemapCS", computeShader);
		}

		{
			RenderBackendShaderDesc computeShaderDesc;
			LoadShaderSourceFromFile("../../../Shaders/ComputeEnviromentIrradiance.hsf", source);
			CompileShader(
				shaderCompiler,
				source,
				HE_TEXT("ComputeEnviromentIrradianceCS"),
				RenderBackendShaderStage::Compute,
				ShaderRepresentation::SPIRV,
				includeDirs,
				defines,
				&computeShaderDesc.stages[(uint32)RenderBackendShaderStage::Compute]);
			computeShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Compute] = "ComputeEnviromentIrradianceCS";
			RenderBackendShaderHandle computeShader = RenderBackendCreateShader(renderBackend, deviceMask, &computeShaderDesc, "ComputeEnviromentIrradianceCS");
			loadedShaders.emplace("ComputeEnviromentIrradianceCS", computeShader);
		}

		{
			RenderBackendShaderDesc computeShaderDesc;
			LoadShaderSourceFromFile("../../../Shaders/FilterEnviromentMap.hsf", source);
			CompileShader(
				shaderCompiler,
				source,
				HE_TEXT("FilterEnviromentMapCS"),
				RenderBackendShaderStage::Compute,
				ShaderRepresentation::SPIRV,
				includeDirs,
				defines,
				&computeShaderDesc.stages[(uint32)RenderBackendShaderStage::Compute]);
			computeShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Compute] = "FilterEnviromentMapCS";
			RenderBackendShaderHandle computeShader = RenderBackendCreateShader(renderBackend, deviceMask, &computeShaderDesc, "FilterEnviromentMapCS");
			loadedShaders.emplace("FilterEnviromentMapCS", computeShader);
		}
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