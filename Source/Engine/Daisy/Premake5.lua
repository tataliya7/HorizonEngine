project "Daisy"
    kind "StaticLib"
    language "C++"
    cppdialect "C++20"
    staticruntime "on"
    location "%{wks.location}/%{prj.name}"
    targetdir "%{wks.location}/Bin/%{cfg.buildcfg}"
        
    links {
        "Core",
        "Render",
        "DxcShaderCompiler",
        "VulkanRenderBackend",
        "HybridRenderPipeline",
    }

    files {
        "**.h",  
        "**.c", 
        "**.hpp",
        "**.cpp",
        "**.inl",
        "**.hsf",
    }
    
    includedirs {
        enginepath(""),
        thirdpartypath("assimp/include"),
        thirdpartypath("entt/include"),
        thirdpartypath("dxc/include"),
        thirdpartypath("glfw/include"),
        thirdpartypath("glm/include"),
        thirdpartypath("spdlog/include"),
        thirdpartypath("vma/include"),
        thirdpartypath("vulkan/include"),
        thirdpartypath("mpmc/include"),
        thirdpartypath("imgui/include"),
    }

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        defines "HORIZON_DEBUG_MODE"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        defines "HORIZON_RELEASE_MODE"
        runtime "Release"
        optimize "on"
