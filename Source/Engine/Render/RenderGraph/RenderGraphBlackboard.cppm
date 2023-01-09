module;

#include <vector>
#include <optional>

#include "Core/CoreDefinitions.h"

export module HorizonEngine.Render.RenderGraph:RenderGraphBlackboard;

import HorizonEngine.Core;
import HorizonEngine.Render.Core;

export namespace HE
{
    class RenderGraphBlackboard
    {
    public:
        RenderGraphBlackboard(MemoryArena* arena) : arena(arena) {}
        RenderGraphBlackboard(RenderGraphBlackboard&& other) = delete;
        RenderGraphBlackboard(const RenderGraphBlackboard& other) = delete;

        template<typename StructType>
        StructType& CreateSingleton()
        {
            const uint32 structIndex = GetStructIndex<StructType>();
            if (structIndex >= (uint32)blackboard.size())
            {
                blackboard.resize(structIndex + 1);
            }

            Struct*& result = blackboard[structIndex];
            ASSERT(!result && "RenderGraphBlackboard duplicate CreateSingleton() called. Only one Register() call per struct is allowed.");
            result = (Struct*)HE_ARENA_ALLOC(arena, sizeof(StructType));
            ASSERT(result);
            return  static_cast<StructTyped<StructType>*>(result)->instance;
        }

        template<typename StructType>
        FORCEINLINE StructType& Get() const
        {
            const uint32 structIndex = GetStructIndex<StructType>();
            ASSERT((structIndex < (uint32)blackboard.size()) && "Failed to find struct instance.");
            StructTyped<StructType>* result = static_cast<StructTyped<StructType>*>(blackboard[structIndex]);
            return result->instance;
        }

        template<typename StructType>
        FORCEINLINE std::optional<StructType>& GetOptional() const
        {
            const uint32 structIndex = GetStructIndex<StructType>();
            if (structIndex < (uint32)blackboard.size())
            {
                return *(static_cast<const StructTyped<StructType>*>(blackboard[structIndex])->instance);
            }
            return std::nullopt;
        }

        void Clear()
        {
            blackboard.clear();
        }

    private:

        struct Struct {};

        template<typename StructType>
        struct StructTyped final : public Struct
        {
            template<typename... Args>
            FORCEINLINE StructTyped(Args&&... args) : instance(std::forward<Args&&>(args)...) {}
            StructType instance;
        };
        static std::string GetStructName(const char* structName, const char* filename, uint32 line);
        static uint32 AllocateIndex(std::string&& structName);

        template<typename StructType>
        static std::string GetStructName()
        {
            static_assert(sizeof(StructType) == 0, "Struct has not been registered with the render graph blackboard. Use RENDER_GRAPH_BLACKBOARD_REGISTER_STRUCT to do this.");
        }

        template<typename StructType>
        static uint32 GetStructIndex()
        {
            static uint32 index = UINT32_MAX;
            if (index == UINT32_MAX)
            {
                index = AllocateIndex(GetStructName<StructType>());
            }
            return index;
        }

        MemoryArena* arena;
        std::vector<Struct*> blackboard;
    };
}
