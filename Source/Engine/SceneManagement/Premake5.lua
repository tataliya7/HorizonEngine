project "SceneManagement"
    kind "StaticLib"
    language "C++"
    cppdialect "C++20"
    staticruntime "on"
    location "%{wks.location}/%{prj.name}"
    targetdir "%{wks.location}/Bin/%{cfg.buildcfg}"
        
    links {
        "Core",
        "ECS",
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
        enginepath(""),
        thirdpartypath("entt/include"),
        thirdpartypath("glm/include"),
        thirdpartypath("yaml-cpp/include"),
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
