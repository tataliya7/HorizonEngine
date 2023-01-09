project "HorizonEditorLauncher"
    kind "ConsoleApp"
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
        "**.cu",
    }

    links {
        "Core",
        "Render",
        "Physics",
        "Entity",
        "Input",
        "Daisy",
        "Scene",
        "VulkanRenderBackend",
        "DxcShaderCompiler",
        "HybridRenderPipeline",
        "AssimpImporter",
        "yaml-cpp",
        "Audio",
        thirdpartypath("glfw/lib/glfw3.lib"),
        thirdpartypath("assimp/lib/assimp-vc143-mt.lib"),
        thirdpartypath("dxc/lib/dxcompiler.lib"),
        thirdpartypath("vulkan/lib/vulkan-1.lib"),
        thirdpartypath("imgui/lib/imgui.lib"),
        thirdpartypath("optick/lib/OptickCore.lib"),
    }
    
    postbuildcommands {
        "{COPY} %{wks.location}/../ThirdParty/assimp/lib/assimp-vc143-mt.dll %{cfg.targetdir}",
        "{COPY} %{wks.location}/../ThirdParty/optick/lib/OptickCore.dll %{cfg.targetdir}",
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
        links {
            thirdpartypath("physx/lib/debug/PhysX_64.lib"),
            thirdpartypath("physx/lib/debug/PhysXCommon_64.lib"),
            thirdpartypath("physx/lib/debug/PhysXFoundation_64.lib"),
            thirdpartypath("physx/lib/debug/PhysXExtensions_static_64.lib"),
            thirdpartypath("physx/lib/debug/PhysXPvdSDK_static_64.lib"),
            thirdpartypath("physx/lib/debug/PhysXVehicle_static_64.lib"),
            thirdpartypath("physx/lib/debug/PhysXVehicle2_static_64.lib"),
            thirdpartypath("physx/lib/debug/PhysXCharacterKinematic_static_64.lib"),
            thirdpartypath("physx/lib/debug/PhysXCooking_64.lib"),
            thirdpartypath("physx/lib/debug/PVDRuntime_64.lib"),
            thirdpartypath("physx/lib/debug/SceneQuery_static_64.lib"),
        }
        postbuildcommands {
            "{COPY} %{wks.location}/../ThirdParty/physx/lib/debug/PhysX_64.dll %{cfg.targetdir}",
            "{COPY} %{wks.location}/../ThirdParty/physx/lib/debug/PhysXFoundation_64.dll %{cfg.targetdir}",
            "{COPY} %{wks.location}/../ThirdParty/physx/lib/debug/PhysXCommon_64.dll %{cfg.targetdir}",
            "{COPY} %{wks.location}/../ThirdParty/physx/lib/debug/PhysXCooking_64.dll %{cfg.targetdir}",
            "{COPY} %{wks.location}/../ThirdParty/physx/lib/debug/PVDRuntime_64.dll %{cfg.targetdir}",
        }

    filter "configurations:Release"
        runtime "Release"
        optimize "on"
        links {
            thirdpartypath("physx/lib/release/PhysX_64.lib"),
            thirdpartypath("physx/lib/release/PhysXCommon_64.lib"),
            thirdpartypath("physx/lib/release/PhysXFoundation_64.lib"),
            thirdpartypath("physx/lib/release/PhysXExtensions_static_64.lib"),
            thirdpartypath("physx/lib/release/PhysXPvdSDK_static_64.lib"),
            thirdpartypath("physx/lib/release/PhysXVehicle_static_64.lib"),
            thirdpartypath("physx/lib/release/PhysXVehicle2_static_64.lib"),
            thirdpartypath("physx/lib/release/PhysXCharacterKinematic_static_64.lib"),
            thirdpartypath("physx/lib/release/PhysXCooking_64.lib"),
            thirdpartypath("physx/lib/release/PVDRuntime_64.lib"),
            thirdpartypath("physx/lib/release/SceneQuery_static_64.lib"),
        }
        postbuildcommands {
            "{COPY} %{wks.location}/../ThirdParty/physx/lib/release/PhysX_64.dll %{cfg.targetdir}",
            "{COPY} %{wks.location}/../ThirdParty/physx/lib/release/PhysXFoundation_64.dll %{cfg.targetdir}",
            "{COPY} %{wks.location}/../ThirdParty/physx/lib/release/PhysXCommon_64.dll %{cfg.targetdir}",
            "{COPY} %{wks.location}/../ThirdParty/physx/lib/release/PhysXCooking_64.dll %{cfg.targetdir}",
            "{COPY} %{wks.location}/../ThirdParty/physx/lib/release/PVDRuntime_64.dll %{cfg.targetdir}",
        }

    filter { "platforms:Win64", "configurations:Debug" }
        linkoptions {"/NODEFAULTLIB:LIBCMT"}
