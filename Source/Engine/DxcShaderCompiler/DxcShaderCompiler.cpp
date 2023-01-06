#include "DxcShaderCompiler.h"

#include <vector>
#include <windows.h>
#include <atlbase.h>
#include <dxc/dxcapi.h>
#include <dxc/d3d12shader.h>

__pragma(warning(push, 0))
import HorizonEngine.Core;
import HorizonEngine.Render.Core;
__pragma(warning(pop))

namespace HE
{
    struct DxcShaderCompiler
    {
        CComPtr<IDxcCompilerArgs> args;
    };

    static bool DxcCompileShader(
        void* instance,
        std::vector<uint8> source,
        const wchar* entry,
        RenderBackendShaderStage stage,
        ShaderRepresentation representation,
        const std::vector<const wchar*>& includeDirs,
        const std::vector<const wchar*>& defines,
        ShaderBlob* outBlob)
    {
        DxcShaderCompiler* dxcShaderCompiler = (DxcShaderCompiler*)instance;
        if (dxcShaderCompiler->args == nullptr)
        {
            return false;
        }

        CComPtr<IDxcUtils> dxcUtils;
        CComPtr<IDxcCompiler3> dxcCompiler;
        HRESULT hr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxcUtils));
        if (FAILED(hr))
        {
            return false;
        }
        hr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler));
        if (FAILED(hr))
        {
            return false;
        }

        static const LPCWSTR targetProfiles[] = {
            TEXT("vs_6_0"),
            TEXT("ps_6_0"),
            TEXT("cs_6_0"),
            L"lib_6_3", // raygen
            L"lib_6_3", // any hit
            L"lib_6_3", // closest hit
            L"lib_6_3", // miss
            L"lib_6_3", // intersection
            L"as_6_5",  // task
            L"ms_6_5",  // mesh
        };

        std::vector<LPCWSTR> args = {};
        switch (representation)
        {
        case ShaderRepresentation::DXIL:
            args.push_back(TEXT("-D")); args.push_back(TEXT("DXIL"));
            break;
        case ShaderRepresentation::SPIRV:
            args.push_back(TEXT("-D")); args.push_back(TEXT("SPIRV"));
            args.push_back(TEXT("-spirv"));
            args.push_back(TEXT("-fspv-target-env=vulkan1.3"));
            args.push_back(TEXT("-fvk-use-scalar-layout"));
            args.push_back(TEXT("-fvk-use-dx-position-w"));
            break;
        default:
            INVALID_ENUM_VALUE();
            return false;
        }

        args.push_back(TEXT("-T"));
        args.push_back(targetProfiles[(uint32)stage]);
        args.push_back(TEXT("-E"));
        args.push_back(entry);

        for (auto& includeDirs : includeDirs)
        {
            args.push_back(TEXT("-I"));
            args.push_back(includeDirs);
        }
        for (auto& define : defines)
        {
            args.push_back(TEXT("-D"));
            args.push_back(define);
        }

        const bool isRaytracingStage = ((stage >= RenderBackendShaderStage::RayGen) && (stage <= RenderBackendShaderStage::Intersection));

        const DxcBuffer buffer = {
            .Ptr = source.data(),
            .Size = source.size(),
            .Encoding = DXC_CP_ACP
        };

        CComPtr<IDxcIncludeHandler> includeHandler;
        hr = dxcUtils->CreateDefaultIncludeHandler(&includeHandler);
        ASSERT(SUCCEEDED(hr));

        CComPtr<IDxcResult> dxcResult;
        hr = dxcCompiler->Compile(&buffer, args.data(), (uint32)args.size(), includeHandler, IID_PPV_ARGS(&dxcResult));
        CComPtr<IDxcBlobUtf8> errors;
        hr = dxcResult->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
        ASSERT(SUCCEEDED(hr));
        if (errors != nullptr && errors->GetStringLength() != 0)
        {
            char msg[1000];
            memcpy(msg, errors->GetStringPointer(), errors->GetStringLength());
            // HE_LOG_ERROR(errors->GetStringPointer());
            printf("%s\n", msg);
        }

        dxcResult->GetStatus(&hr);
        if (SUCCEEDED(hr))
        {
            CComPtr<IDxcBlob> blob;
            hr = dxcResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&blob), nullptr);
            ASSERT(SUCCEEDED(hr));
            outBlob->size = blob->GetBufferSize();
            outBlob->data = (uint8*)malloc(outBlob->size);
            memcpy(outBlob->data, blob->GetBufferPointer(), outBlob->size);
        }
        return SUCCEEDED(hr);
    }

    static void DxcReleaseShaderBlob(void* instance, ShaderBlob* blob)
    {
        delete(blob->data);
    }

    ShaderCompiler* CreateDxcShaderCompiler()
    {
        CComPtr<IDxcCompilerArgs> args;
        HRESULT hr = DxcCreateInstance(CLSID_DxcCompilerArgs, IID_PPV_ARGS(&args));
        if (FAILED(hr))
        {
            return nullptr;
        }

        static const WCHAR* compilerArgs[] = { TEXT("-P") };
        static const DxcDefine compilerDefines[] = { { TEXT("DXC"), nullptr} };
        hr = args->AddArguments(compilerArgs, ARRAY_SIZE(compilerArgs));
        ASSERT(SUCCEEDED(hr));
        hr = args->AddDefines(compilerDefines, ARRAY_SIZE(compilerDefines));
        ASSERT(SUCCEEDED(hr));

        ShaderCompiler* compiler = new ShaderCompiler();
        DxcShaderCompiler* dxcCompiler = new DxcShaderCompiler();
        dxcCompiler->args = args;

        compiler->instance = dxcCompiler;
        compiler->CompileShader = DxcCompileShader;
        compiler->ReleaseShaderBlob = DxcReleaseShaderBlob;

        return compiler;
    }

    void DestroyDxcShaderCompiler(ShaderCompiler* compiler)
    {
        DxcShaderCompiler* dxcCompiler = (DxcShaderCompiler*)compiler->instance;
        if (dxcCompiler->args != nullptr)
        {
            dxcCompiler->args.Release();
        }
        delete dxcCompiler;
        delete compiler;
    }
}