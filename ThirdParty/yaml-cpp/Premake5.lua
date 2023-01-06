project "yaml-cpp"
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
        thirdpartypath("yaml-cpp/include"),
    }

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        runtime "Release"
        optimize "on"
