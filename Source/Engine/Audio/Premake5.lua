project "Audio"
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
        "**.cppm",
        "**.inl",
        "**.hsf",
    }

    includedirs {
        "",
        enginepath(""),
        thirdpartypath("wwise/include"),
    }

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        defines "HE_DEBUG_MODE"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        defines "HE_DEBUG_MODE"
        runtime "Release"
        optimize "on"
