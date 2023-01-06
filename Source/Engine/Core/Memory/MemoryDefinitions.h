#pragma once

#define HE_ARENA_ALLOC(arena, size)                                          (::HE::ArenaRealloc(arena, nullptr, 0,       size,    HE_DEFAULT_ALIGNMENT, __FILE__, __LINE__))
#define HE_ARENA_FREE(arena, ptr, size)                                      (::HE::ArenaRealloc(arena, ptr,     size,    0,       HE_DEFAULT_ALIGNMENT, __FILE__, __LINE__))
#define HE_ARENA_REALLOC(arena, ptr, oldSize, newSize)                       (::HE::ArenaRealloc(arena, ptr,     oldSize, newSize, HE_DEFAULT_ALIGNMENT, __FILE__, __LINE__))
#define HE_ARENA_ALIGNED_ALLOC(arena, size, alignment)                       (::HE::ArenaRealloc(arena, nullptr, 0,       size,    alignment,            __FILE__, __LINE__))
#define HE_ARENA_ALIGNED_FREE(arena, ptr, size, alignment)                   (::HE::ArenaRealloc(arena, ptr,     size,    0,       alignment,            __FILE__, __LINE__))
#define HE_ARENA_ALIGNED_REALLOC(arena, ptr, oldSize, newSize, alignment)    (::HE::ArenaRealloc(arena, ptr,     oldSize, newSize, alignment,            __FILE__, __LINE__))
