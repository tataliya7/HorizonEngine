project "HorizonEditor"
    kind "StaticLib"
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
        "**.cppm",
        "**.inl",
        "**.hsf",
    }
    dependson {
        "HorizonEngine",
    }

    links {
    }

    includedirs { 
        editorpath(""),
        enginepath(""),
        thirdpartypath("glm/include"),
        thirdpartypath("glfw/include"),
        thirdpartypath("spdlog/include"),   
        thirdpartypath("imgui/include"),
    }

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        runtime "Release"
        optimize "on"
