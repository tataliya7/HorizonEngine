#pragma once

namespace HE
{
	struct ShaderCompiler;
	extern ShaderCompiler* CreateDxcShaderCompiler();
	extern void DestroyDxcShaderCompiler(ShaderCompiler* compiler);
}