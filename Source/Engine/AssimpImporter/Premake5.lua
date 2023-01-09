project "AssimpImporter"
    kind "StaticLib"
    language "C++"
    cppdialect "C++20"
    staticruntime "on"
    location "%{wks.location}/%{prj.name}"
    targetdir "%{wks.location}/Bin/%{cfg.buildcfg}"
        
    links {
        "Core",
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
        enginepath("Core"),
        thirdpartypath("assimp/include"),
        thirdpartypath("spdlog/include"),
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
