project "ModelViewer"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++20"
    staticruntime "on"
    location "%{wks.location}/%{prj.name}"
    targetdir "%{wks.location}/Bin/%{cfg.buildcfg}"

    files {
        "**.h",
        "**.c", 
        "**.hpp",
        "**.cpp",
        "**.inl",
        "**.hsf",
    }

    dependson {
    }
    
    links {
        "Core", 
        "ECS",
        "Input",
        "VulkanRenderBackend",
        "DxcShaderCompiler",
        "RenderEngine",
        "RenderGraph",
        "HybridRenderPipeline",
        "AssimpImporter",
        thirdpartypath("glfw/lib/glfw3.lib"),
        thirdpartypath("assimp/lib/assimp-vc143-mt.lib"),
        thirdpartypath("dxc/lib/dxcompiler.lib"),
        thirdpartypath("vulkan/lib/vulkan-1.lib"),
        thirdpartypath("imgui/lib/imgui.lib"),
    }
    
    postbuildcommands {
        "{COPY} %{wks.location}/../ThirdParty/assimp/bin/assimp-vc143-mt.dll %{cfg.targetdir}",
    }

    includedirs {
        enginepath(""),
        editorpath(""),
        thirdpartypath("glm/include"),
        thirdpartypath("glfw/include"),
        thirdpartypath("spdlog/include"),
        thirdpartypath("imgui/include"), 
        thirdpartypath("entt/include"),
    }

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        runtime "Release"
        optimize "on"

    filter { "platforms:Win64", "configurations:Debug" }
        linkoptions {"/NODEFAULTLIB:LIBCMT"}