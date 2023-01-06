module;

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>

module HorizonEngine.Render.UI;

namespace HE
{
    bool UIRenderer::Init()
    {
        IMGUI_CHECKVERSION();
        context = ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        IM_ASSERT(io.BackendRendererUserData == NULL && "Already initialized a renderer backend!");
        io.BackendRendererName = "Horizon Engine";
        io.ConfigFlags  |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
        io.ConfigFlags  |= ImGuiConfigFlags_DockingEnable;           // Enable Docking

        ImGui::StyleColorsDark();
        ImGuiStyle& style = ImGui::GetStyle();

        // light style from Pacôme Danhiez (user itamago) https://github.com/ocornut/imgui/pull/511#issuecomment-175719267
        style.WindowRounding = 2.0f;
        style.ScrollbarRounding = 3.0f;
        style.GrabRounding = 2.0f;
        style.AntiAliasedLines = true;
        style.AntiAliasedFill = true;
        style.WindowRounding = 2;
        style.ChildRounding = 2;
        style.ScrollbarSize = 16;
        style.ScrollbarRounding = 3;
        style.GrabRounding = 2;
        style.ItemSpacing.x = 10;
        style.ItemSpacing.y = 4;
        style.IndentSpacing = 22;
        style.FramePadding.x = 6;
        style.FramePadding.y = 4;
        style.Alpha = 1.0f;
        style.FrameRounding = 3.0f;

        style.Colors[ImGuiCol_Text] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
        style.Colors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 0.94f);
        style.Colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        style.Colors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
        style.Colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
        style.Colors[ImGuiCol_BorderShadow] = ImVec4(1.00f, 1.00f, 1.00f, 0.10f);
        style.Colors[ImGuiCol_FrameBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
        style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
        style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        style.Colors[ImGuiCol_TitleBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
        style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
        style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.69f, 0.69f, 0.69f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.49f, 0.49f, 0.49f, 1.00f);
        //style.Colors[ImGuiCol_ComboBg] = ImVec4(0.86f, 0.86f, 0.86f, 0.99f);
        style.Colors[ImGuiCol_CheckMark] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.24f, 0.52f, 0.88f, 1.00f);
        style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
        style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
        style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
        style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_Separator] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
        style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
        style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.50f);
        style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
        //style.Colors[ImGuiCol_CloseButton] = ImVec4(0.59f, 0.59f, 0.59f, 0.50f);
        //style.Colors[ImGuiCol_CloseButtonHovered] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
        //style.Colors[ImGuiCol_CloseButtonActive] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
        style.Colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
        style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
        style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);

        ImGui_ImplGlfw_InitForVulkan((GLFWwindow*)window, true);

        // Upload Fonts
        {
            io.FontDefault = io.Fonts->AddFontFromFileTTF("../../../Assets/Fonts/OpenSans/OpenSans-Regular.ttf", 16.0f);
            unsigned char* pixels;
            int width, height;
            io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
            RenderBackendTextureDesc defaultFontTextureDesc = RenderBackendTextureDesc::Create2D(
                width,
                height,
                PixelFormat::RGBA8Unorm,
                TextureCreateFlags::ShaderResource);
            defaultFontTexture = RenderBackendCreateTexture(renderBackend, deviceMask, &defaultFontTextureDesc, pixels, "DefaultFont");
            io.Fonts->SetTexID((ImTextureID)(uint64)defaultFontTexture);
        }

        RenderBackendShaderDesc imguiShaderDesc;
        imguiShaderDesc.rasterizationState.cullMode = RasterizationCullMode::None;
        imguiShaderDesc.rasterizationState.frontFaceCounterClockwise = true;
        imguiShaderDesc.colorBlendState.numColorAttachments = 1;
        imguiShaderDesc.colorBlendState.attachmentStates[0].blendEnable = true;
        imguiShaderDesc.colorBlendState.attachmentStates[0].srcColorBlendFactor = BlendFactor::SrcAlpha;
        imguiShaderDesc.colorBlendState.attachmentStates[0].dstColorBlendFactor = BlendFactor::OneMinusSrcAlpha;
        imguiShaderDesc.colorBlendState.attachmentStates[0].colorBlendOp = BlendOp::Add;
        imguiShaderDesc.colorBlendState.attachmentStates[0].srcAlphaBlendFactor = BlendFactor::One;
        imguiShaderDesc.colorBlendState.attachmentStates[0].dstAlphaBlendFactor = BlendFactor::OneMinusSrcAlpha;
        imguiShaderDesc.colorBlendState.attachmentStates[0].alphaBlendOp = BlendOp::Add;
        imguiShaderDesc.colorBlendState.attachmentStates[0].colorWriteMask = ColorComponentFlags::All;
        imguiShaderDesc.depthStencilState = {
            .depthTestEnable = false,
            .depthWriteEnable = false,
            .depthCompareOp = CompareOp::Never,
            .stencilTestEnable = false,
        };
        std::vector<uint8> source;
        std::vector<const wchar*> includeDirs;
        std::vector<const wchar*> defines;
        includeDirs.push_back(HE_TEXT("../../../Shaders"));
        LoadShaderSourceFromFile("../../../Shaders/ImGui.hsf", source);
        CompileShader(
            shaderCompiler,
            source,
            HE_TEXT("ImGuiVS"),
            RenderBackendShaderStage::Vertex,
            ShaderRepresentation::SPIRV,
            includeDirs,
            defines,
            &imguiShaderDesc.stages[(uint32)RenderBackendShaderStage::Vertex]);
        imguiShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Vertex] = "ImGuiVS";
        CompileShader(
            shaderCompiler,
            source,
            HE_TEXT("ImGuiPS"),
            RenderBackendShaderStage::Pixel,
            ShaderRepresentation::SPIRV,
            includeDirs,
            defines,
            &imguiShaderDesc.stages[(uint32)RenderBackendShaderStage::Pixel]);

        imguiShaderDesc.entryPoints[(uint32)RenderBackendShaderStage::Pixel] = "ImGuiPS";
        imguiShader = RenderBackendCreateShader(renderBackend, deviceMask, &imguiShaderDesc, "ImGuiPS");

        RenderBackendBufferDesc bufferDesc = RenderBackendBufferDesc::CreateByteAddress(100);
        vertexBuffer = RenderBackendCreateBuffer(renderBackend, deviceMask, &bufferDesc, "ImGuiVertexBuffer");
        indexBuffer = RenderBackendCreateBuffer(renderBackend, deviceMask, &bufferDesc, "ImGuiIndexBuffer");
        return true;
    }

    void UIRenderer::Shutdown()
    {
    }

    void UIRenderer::BeginFrame()
    {
        ImGui::SetCurrentContext(context);
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void UIRenderer::EndFrame()
    {
        ImGui::EndFrame();
        ImGui::Render();

        ImDrawData* drawData = ImGui::GetDrawData();
        // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer coordinates)
        int fbWidth = (int)(drawData->DisplaySize.x * drawData->FramebufferScale.x);
        int fbHeight = (int)(drawData->DisplaySize.y * drawData->FramebufferScale.y);
        if (fbWidth <= 0 || fbHeight <= 0)
        {
            return;
        }

        if (drawData->TotalVtxCount > 0)
        {
            // Create or reserve the vertex/index buffers
            uint64 newVertexBufferSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
            uint64 newIndexBufferSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);
            if (vertexBufferSize < newVertexBufferSize)
            {
                RenderBackendResizeBuffer(renderBackend, vertexBuffer, newVertexBufferSize);
                vertexBufferSize = newVertexBufferSize;
            }
            if (indexBufferSize < newIndexBufferSize)
            {
                RenderBackendResizeBuffer(renderBackend, indexBuffer, newIndexBufferSize);
                indexBufferSize = newIndexBufferSize;
            }
            vertices.reserve(drawData->TotalVtxCount);
            indices.reserve(drawData->TotalIdxCount);
            uint32 vertexOffset = 0;
            uint32 indexOffset = 0;
            for (int i = 0; i < drawData->CmdListsCount; i++)
            {
                const ImDrawList* cmdList = drawData->CmdLists[i];
                memcpy(vertices.data() + vertexOffset, cmdList->VtxBuffer.Data, cmdList->VtxBuffer.Size * sizeof(ImDrawVert));
                memcpy(indices.data() + indexOffset, cmdList->IdxBuffer.Data, cmdList->IdxBuffer.Size * sizeof(ImDrawIdx));
                vertexOffset += cmdList->VtxBuffer.Size;
                indexOffset += cmdList->IdxBuffer.Size;
            }
            
            void* data = nullptr;
            RenderBackendMapBuffer(renderBackend, vertexBuffer, &data);
            memcpy((uint8*)data, vertices.data(), vertexBufferSize);
            RenderBackendUnmapBuffer(renderBackend, vertexBuffer);

            RenderBackendMapBuffer(renderBackend, indexBuffer, &data);
            memcpy((uint8*)data, indices.data(), indexBufferSize);
            RenderBackendUnmapBuffer(renderBackend, indexBuffer);
        }
    }

    void UIRenderer::Render(RenderCommandList& commandList, RenderBackendTextureHandle outputTexture)
    {
        ImDrawData* drawData = ImGui::GetDrawData();
        // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer coordinates)
        int fbWidth = (int)(drawData->DisplaySize.x * drawData->FramebufferScale.x);
        int fbHeight = (int)(drawData->DisplaySize.y * drawData->FramebufferScale.y);
        if (fbWidth <= 0 || fbHeight <= 0)
        {
            return;
        }

        RenderBackendViewport viewport(0.0f, (float)fbHeight, (float)fbWidth, -(float)fbHeight);

        commandList.SetViewports(&viewport, 1);

        RenderPassInfo renderPass = { 
            .colorRenderTargets = { {.texture = outputTexture, .mipLevel = 0, .arrayLayer = 0, .loadOp = RenderTargetLoadOp::Load, .storeOp = RenderTargetStoreOp::Store } },
        };
        commandList.BeginRenderPass(renderPass);

        // Will project scissor/clipping rectangles into framebuffer space
        ImVec2 clipOffset = drawData->DisplayPos;         // (0,0) unless using multi-viewports
        ImVec2 clipScale = drawData->FramebufferScale;    // (1,1) unless using retina display which are often (2,2)

        // Render command lists
        // (Because we merged all buffers into a single one, we maintain our own offset into them)
        int globalIndexOffset = 0;
        int globalVertexOffset = 0;
        for (int i = 0; i < drawData->CmdListsCount; i++)
        {
            const ImDrawList* cmdList = drawData->CmdLists[i];
            for (int drawCallIndex = 0; drawCallIndex < cmdList->CmdBuffer.Size; drawCallIndex++)
            {
                const ImDrawCmd* pcmd = &cmdList->CmdBuffer[drawCallIndex];

                // Project scissor/clipping rectangles into framebuffer space
                ImVec2 clipMin((pcmd->ClipRect.x - clipOffset.x) * clipScale.x, (pcmd->ClipRect.y - clipOffset.y) * clipScale.y);
                ImVec2 clipMax((pcmd->ClipRect.z - clipOffset.x) * clipScale.x, (pcmd->ClipRect.w - clipOffset.y) * clipScale.y);

                // Clamp to viewport as vkCmdSetScissor() won't accept values that are off bounds
                if (clipMin.x < 0.0f) { clipMin.x = 0.0f; }
                if (clipMin.y < 0.0f) { clipMin.y = 0.0f; }
                if (clipMax.x > fbWidth) { clipMax.x = (float)fbWidth; }
                if (clipMax.y > fbHeight) { clipMax.y = (float)fbHeight; }
                if (clipMax.x <= clipMin.x || clipMax.y <= clipMin.y)
                {
                    continue;
                }

                RenderBackendScissor scissor((int32)(clipMin.x), (int32)(clipMin.y), (uint32)(clipMax.x - clipMin.x), (uint32)(clipMax.y - clipMin.y));
                commandList.SetScissors(&scissor, 1);

                float scale[2];
                scale[0] = 2.0f / drawData->DisplaySize.x;
                scale[1] = 2.0f / drawData->DisplaySize.y;
                float translate[2];
                translate[0] = -1.0f - drawData->DisplayPos.x * scale[0];
                translate[1] = -1.0f - drawData->DisplayPos.y * scale[1];

                ShaderArguments shaderArguments = {};
                shaderArguments.BindTextureSRV(0, RenderBackendTextureSRVDesc::Create(defaultFontTexture));
                shaderArguments.BindBuffer(1, vertexBuffer, pcmd->VtxOffset + globalVertexOffset);
                // shaderArguments.BindBuffer(2, indexBuffer, pcmd->IdxOffset + globalIndexOffset);
                shaderArguments.PushConstants(0, scale[0]);
                shaderArguments.PushConstants(1, scale[1]);
                shaderArguments.PushConstants(2, translate[0]);
                shaderArguments.PushConstants(3, translate[1]);

                commandList.DrawIndexed(
                    imguiShader,
                    shaderArguments, 
                    indexBuffer,
                    pcmd->ElemCount,
                    1,
                    pcmd->IdxOffset + globalIndexOffset,
                    pcmd->VtxOffset + globalVertexOffset,
                    0, 
                    PrimitiveTopology::TriangleList);
            }
            globalIndexOffset += cmdList->IdxBuffer.Size;
            globalVertexOffset += cmdList->VtxBuffer.Size;
        }
        commandList.EndRenderPass();
    }
}