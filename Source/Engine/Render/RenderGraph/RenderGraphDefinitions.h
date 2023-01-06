#pragma once

#define RENDER_GRAPH_BLACKBOARD_REGISTER_STRUCT(StructType)				           \
template<>															               \
FORCEINLINE std::string RenderGraphBlackboard::GetStructName<StructType>()         \
{																	               \
	return GetStructName(#StructType, __FILE__, __LINE__);                         \
}

/**
 *
 * Example of usage:
 *
 * struct MyStruct
 * {
 *     RenderGraphTextureHandle texture;
 * };
 * RENDER_GRAPH_BLACKBOARD_REGISTER_STRUCT(MyStruct);
 *
 * void Func1(RenderGraphBlackboard& blackboard)
 * {
 *     auto& myStruct = blackboard.CreateSingleton<MyStruct>();
 *     // ...
 * }
 *
 * void Func2(RenderGraphBlackboard& blackboard)
 * {
 *     const auto& myStruct = blackboard.Get<MyStruct>();
 *     // ...
 * }
 *
 */