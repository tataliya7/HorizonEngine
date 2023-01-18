# Horizon Engine

Horizon Engine is an open-source 3D rendering engine, focusing on modern rendering engine architecture and rendering techniques. Serving a different purpose than game engines, this project aims to build a highly scalable rendering framework to improve the productivity of prototype projects and academic research, but also to serve as an educational tool for teaching rendering engine design and implementation from scratch.

Horizon Engine is currently only supported on Windows and only target modern graphics APIs (Direct3D 12, Vulkan, Metal).

<!--
[![Bilibili]()]()
-->
[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/nepzQHf2jv)
<!--
# [![Patreon](https://badgen.net/badge/icon/patreon?icon=patreon&label)]()
-->

## Introduction

Goals:
* Efficent and Flexible Rendering
* Fast Rendering Techniques Experimentation, eg. Hybrid Rendering with DXR or Vulkan Ray Tracing KHR

Horizon Engine Architecture:

![image](/Docs/engine_architecture.png)

Daisy Renderer Architecture:

Inspired by [Halcyon](https://www.ea.com/seed/news/khronos-munich-2018-halcyon-vulkan)

![image](/Docs/daisy_architecture.png)

Features:
 - [x] Modularization using C++20 modules
 - [x] Fiber-Based Job System
 - [x] Bindless
 - [x] Render Graph
 - [x] ECS (Entity-Component-System)
 - [x] Hybrid Render Pipeline
 - [x] Deferred Render Pipeline
 - [x] Path Tracing
 - [x] Physics Engine Integration (PhysX)
 - [x] Sound Engine Integration (miniaudio)

Roadmap:
 - [ ] Super Resolution Algorithms (DLSS 2, FSR 2, etc)
 - [ ] Visibility Buffer (Also Called Deferred Texturing or Deferred Material Shading)
 - [ ] Surfle GI
 - [ ] Native Graphics API Support (Direct3D 12, Metal 2)
 - [ ] Multi-GPU Rendering
 - [ ] Plugin System (Live-Reloadable Plugins)
 - [ ] Machine Learning (TensorRT Integration)
 - [ ] VR/AR Support
 - [ ] Cross-Platform (MacOS and Linux Support)

## Screenshots

Scene Editor
![image](/Screenshots/screenshot_editor.png)

FSR
![image](/Screenshots/screenshot_fsr.png)

Raytraced Shadows
![image](/Screenshots/screenshot_raytraced_shadow.jpg)

Sky Atomsphere
![image](/Screenshots/screenshot_sky_atmosphere.png)

Particles
![image](/Screenshots/screenshot_particle_system.png)

Cornell Box
![image](/Screenshots/screenshot_cornell_box.png)

See [here]() for more examples.

## Requirements

* Windows 10 SDK (10.0.20348.0)
* NIVIDIA Graphics Cards (Geforce RTX 20-series or 30-series) and keep your graphics drivers up to date (https://www.nvidia.com/Download/index.aspx)
* Vulkan SDK 1.3.216.0, this repository tries to always be up to date with the latest Vulkan SDK (https://vulkan.lunarg.com/sdk/home)
* CUDA SDK (11.7)
* Visual Studio 2022, and make sure C++ Modules for v143 build tools (x64/x86 - experimental) is installed (Currently, I'm working with Preview Version 17.4.0, it may not compile successfully if you use another version)
* C++ 20

## Getting Started

1. Clone this repository

`git clone https://github.com/harukumo/HorizonEngine`

2. Run "GenerateProjects.bat"

## Third Party

* [assimp](https://github.com/assimp/assimp)
* [imgui](https://github.com/ocornut/imgui)
* [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo)
* [entt](https://github.com/skypjack/entt)
* [glfw](https://github.com/glfw/glfw)
* [stb](https://github.com/nothings/stb)
* [spdlog](https://github.com/gabime/spdlog)
* [glm](https://github.com/g-truc/glm)
* [Vulkan](https://www.khronos.org/vulkan)
* [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
* [DirectX-Headers](https://github.com/microsoft/DirectX-Headers)
* [DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler)
* [D3D12MemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator)
* [OpenEXR](https://github.com/AcademySoftwareFoundation/openexr)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp)
* [MPMCQueue](https://github.com/rigtorp/MPMCQueue)
* [PhysX](https://github.com/NVIDIAGameWorks/PhysX)
* [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix)
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
* [premake](https://github.com/premake/premake-core)
* [optick](https://github.com/bombomby/optick)
* [miniaudio](https://github.com/mackron/miniaudio)

## Documentation

[Reference Manual](https://harukumo.github.io/HorizonEngine/)

API Specification

Programming Guide

Q&A:

See [Q&A](/Docs/Q&A.md) where common questions are answered.

## Debug Tools

Vulkan/D3D12 debugging: 
* [RenderDoc](https://renderdoc.org/), but it dose not support Vulkan Ray Tracing KHR or DXR
* Use [NVIDIA Nsight Graphics](https://developer.nvidia.com/nsight-graphics) if ray tracing is enabled

Optix debugging:
* [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)

CPU profiling: 
* [Optick](https://github.com/bombomby/optick)

## Contribution

Contributions are welcome.

## Sponsors


## Lisence