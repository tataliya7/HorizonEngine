project "ExampleBase"
    kind "StaticLib"
    language "C++"
    cppdialect "C++20"
    staticruntime "on"
    location "%{wks.location}/%{prj.name}"
    targetdir "%{wks.location}/Bin/%{cfg.buildcfg}"
    debugdir "%{cfg.targetdir}"

    files {
        "**.h",
        "**.c", 
        "**.hpp",
        "**.cpp",
        "**.cppm",
        "**.inl",
        "**.hsf",
    }

    links {
        "Core",
        "ECS",
        "Render",
        "Physics",
        "Input",
        "DxcShaderCompiler",
        "VulkanRenderBackend",
        "HybridRenderPipeline",
    }

    includedirs {
        enginepath(""),
        editorpath(""),
        thirdpartypath("glm/include"),
        thirdpartypath("glfw/include"),
        thirdpartypath("spdlog/include"),
        thirdpartypath("imgui/include"), 
        thirdpartypath("entt/include"),
        thirdpartypath("optick/include"),
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