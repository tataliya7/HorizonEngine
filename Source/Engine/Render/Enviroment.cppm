module;

#include "Core/CoreDefinitions.h"
#include "Render/RenderDefinitions.h"

export module HorizonEngine.Render.Enviroment;

import HorizonEngine.Core;
import HorizonEngine.Render.Core;
import HorizonEngine.Render.ShaderSystem;

export namespace HE
{
	uint32 GIrradianceEnviromentMapSize = 32;

	void GenerateCubemapMips(RenderCommandList& commandList, RenderBackendTextureHandle cubemap, uint32 numMipLevels)
	{
		ShaderLibrary* shaderLibrary = GetGlobalShaderLibrary();
		RenderBackendShaderHandle computeShader = shaderLibrary->GetShader("DownsampleCubemapCS");

		for (uint32 mipLevel = 1; mipLevel < numMipLevels; mipLevel++)
		{
			RenderBackendBarrier transitions[] =
			{
				RenderBackendBarrier(cubemap, RenderBackendTextureSubresourceRange(mipLevel - 1, 1, 0, REMAINING_ARRAY_LAYERS), RenderBackendResourceState::UnorderedAccess, RenderBackendResourceState::ShaderResource),
				RenderBackendBarrier(cubemap, RenderBackendTextureSubresourceRange(mipLevel, 1, 0, REMAINING_ARRAY_LAYERS), RenderBackendResourceState::Undefined, RenderBackendResourceState::UnorderedAccess)
			};
			commandList.Transitions(transitions, 2);

			uint32 dispatchX = CEIL_DIV(1 << (numMipLevels - mipLevel - 1), 8);
			uint32 dispatchY = CEIL_DIV(1 << (numMipLevels - mipLevel - 1), 8);
			uint32 dispatchZ = 1;

			ShaderArguments shaderArguments = {};
			shaderArguments.BindTextureSRV(0, RenderBackendTextureSRVDesc::Create(cubemap));
			shaderArguments.BindTextureUAV(1, RenderBackendTextureUAVDesc::Create(cubemap, mipLevel));
			shaderArguments.PushConstants(0, (float)(mipLevel - 1));

			commandList.Dispatch(
				computeShader,
				shaderArguments,
				dispatchX,
				dispatchY,
				dispatchZ);
		}

		RenderBackendBarrier transition = RenderBackendBarrier(cubemap, RenderBackendTextureSubresourceRange(numMipLevels - 1, REMAINING_MIP_LEVELS, 0, REMAINING_ARRAY_LAYERS), RenderBackendResourceState::UnorderedAccess, RenderBackendResourceState::ShaderResource);
		commandList.Transitions(&transition, 1);
	}

	void ComputeEnviromentIrradiance(RenderCommandList& commandList, RenderBackendTextureHandle enviromentMap, uint32 mipLevel, RenderBackendTextureHandle irradianceEnviromentMap)
	{
		ShaderLibrary* shaderLibrary = GetGlobalShaderLibrary();
		RenderBackendShaderHandle computeShader = shaderLibrary->GetShader("ComputeEnviromentIrradianceCS");

		RenderBackendBarrier transition(irradianceEnviromentMap, RenderBackendTextureSubresourceRange(0, 1, 0, 6), RenderBackendResourceState::Undefined, RenderBackendResourceState::UnorderedAccess);
		commandList.Transitions(&transition, 1);

		uint32 dispatchX = CEIL_DIV(GIrradianceEnviromentMapSize, 8);
		uint32 dispatchY = CEIL_DIV(GIrradianceEnviromentMapSize, 8);
		uint32 dispatchZ = 1;

		ShaderArguments shaderArguments = {};
		shaderArguments.BindTextureSRV(0, RenderBackendTextureSRVDesc::Create(enviromentMap));
		shaderArguments.BindTextureUAV(1, RenderBackendTextureUAVDesc::Create(irradianceEnviromentMap, 0));
		shaderArguments.PushConstants(0, (float)mipLevel);

		commandList.Dispatch(
			computeShader,
			shaderArguments,
			dispatchX,
			dispatchY,
			dispatchZ);

		transition = RenderBackendBarrier(irradianceEnviromentMap, RenderBackendTextureSubresourceRange(0, 1, 0, 6), RenderBackendResourceState::UnorderedAccess, RenderBackendResourceState::ShaderResource);
		commandList.Transitions(&transition, 1);
	}

	void FilterEnviromentMap(RenderCommandList& commandList, RenderBackendTextureHandle enviromentMap, uint32 numMipLevels, RenderBackendTextureHandle filteredEnviromentMap)
	{
		ShaderLibrary* shaderLibrary = GetGlobalShaderLibrary();
		RenderBackendShaderHandle computeShader = shaderLibrary->GetShader("FilterEnviromentMapCS");

		RenderBackendBarrier transition(filteredEnviromentMap, RenderBackendTextureSubresourceRange(0, REMAINING_MIP_LEVELS, 0, REMAINING_ARRAY_LAYERS), RenderBackendResourceState::Undefined, RenderBackendResourceState::UnorderedAccess);
		commandList.Transitions(&transition, 1);

		for (uint32 mipLevel = 0; mipLevel < numMipLevels; mipLevel++)
		{
			uint32 dispatchX = CEIL_DIV(1 << (numMipLevels - mipLevel - 1), 8);
			uint32 dispatchY = CEIL_DIV(1 << (numMipLevels - mipLevel - 1), 8);
			uint32 dispatchZ = 1;

			float roughness = (float)mipLevel / (float)(numMipLevels - 1);

			ShaderArguments shaderArguments = {};
			shaderArguments.BindTextureSRV(0, RenderBackendTextureSRVDesc::Create(enviromentMap));
			shaderArguments.BindTextureUAV(1, RenderBackendTextureUAVDesc::Create(filteredEnviromentMap, mipLevel));
			shaderArguments.PushConstants(0, roughness);

			commandList.Dispatch(
				computeShader,
				shaderArguments,
				dispatchX,
				dispatchY,
				dispatchZ);
		}

		transition = RenderBackendBarrier(filteredEnviromentMap, RenderBackendTextureSubresourceRange(0, REMAINING_MIP_LEVELS, 0, REMAINING_ARRAY_LAYERS), RenderBackendResourceState::UnorderedAccess, RenderBackendResourceState::ShaderResource);
		commandList.Transitions(&transition, 1);
	}

	void ComputeEnviromentCubemaps(RenderCommandList& commandList, RenderBackendTextureHandle enviromentMap, uint32 cubemapSize, RenderBackendTextureHandle irradianceEnviromentMap, RenderBackendTextureHandle filteredEnviromentMap)
	{
		const uint32 numMipLevels = Math::MaxNumMipLevels(cubemapSize);

		GenerateCubemapMips(commandList, enviromentMap, numMipLevels);

		const uint32 numIrradianceEnviromentMapMipLevels = Math::MaxNumMipLevels(GIrradianceEnviromentMapSize);
		const uint32 sourceMipLevel = Math::Max<uint32>(0, numMipLevels - numIrradianceEnviromentMapMipLevels);

		ComputeEnviromentIrradiance(commandList, enviromentMap, sourceMipLevel, irradianceEnviromentMap);

		FilterEnviromentMap(commandList, enviromentMap, numMipLevels, filteredEnviromentMap);
	}
}