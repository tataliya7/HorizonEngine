include "AssimpImporter"
include "Core"
include "D3D12RenderBackend"
include "HybridRenderPipeline"
include "Daisy"
include "DxcShaderCompiler"
include "ECS"
include "Render"
include "PhysicS"
include "Audio"
include "Script"
include "ShaderSystem"
include "VulkanRenderBackend"
include "Input"
include "MetaParser"
include "Animation"
include "ParticleSystem"
include "SceneManagement"

project "HorizonEngine"
    kind "StaticLib"
    language "C++"
    cppdialect "C++20"
    staticruntime "on"
    location "%{wks.location}/%{prj.name}"
    targetdir "%{wks.location}/Bin/%{cfg.buildcfg}"
    files {
        "HorizonEngine.cppm"
    }