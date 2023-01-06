project "PathTracing"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++20"
    staticruntime "on"
    location "%{wks.location}/%{prj.name}"
    targetdir "%{wks.location}/Bin/%{cfg.buildcfg}"
    
    links {
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/lib/x64/cudart_static.lib",
    }
    
    files {
        "**.h",  
        "**.c", 
        "**.hpp",
        "**.cpp",
        "**.inl",
        "**.ini",
        "**.cu",
    }

    includedirs {
        "Source",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/include",
        thirdpartypath("optix/include"),
        thirdpartypath("glm/include"),
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
