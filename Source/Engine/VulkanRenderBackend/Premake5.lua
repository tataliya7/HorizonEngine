project "VulkanRenderBackend"
    kind "StaticLib"
    language "C++"
    cppdialect "C++20"
    staticruntime "on"
    location "%{wks.location}/%{prj.name}"
    targetdir "%{wks.location}/Bin/%{cfg.buildcfg}"
        
    links {
        "Core",
        "Render",
    }

    files {
        "**.h",  
        "**.c", 
        "**.hpp",
        "**.cpp",
        "**.cppm",
        "**.inl",
        "**.hsf",
    }

    includedirs {
        "",
        enginepath(""),
        thirdpartypath("vulkan/include"),
        thirdpartypath("vma/include"),
        thirdpartypath("glm/include"),
    }

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        defines "HE_DEBUG_MODE"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        defines "HE_RELEASE_MODE"
        runtime "Release"
        optimize "on"
