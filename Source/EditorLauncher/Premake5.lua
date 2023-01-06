project "EditorLauncher"
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
    }

    includedirs {
        enginepath(""),
        editorpath(""),
        thirdpartypath("glm/include"),
        thirdpartypath("glfw/include"),
    }

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        runtime "Release"
        optimize "on"
