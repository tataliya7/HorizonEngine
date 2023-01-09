module;

#include "CoreCommon.h"

export module HorizonEngine.Core.Memory;

import HorizonEngine.Core.Types;

export namespace HE
{
    class MemoryArena
    {
    public:
        MemoryArena() = default;
        virtual ~MemoryArena() = default;
        MemoryArena(const MemoryArena&) = delete;
        MemoryArena& operator=(const MemoryArena&) = delete;
        virtual void* Alloc(uint64 size, uint64 alignment) = 0;
        virtual void Free(void* ptr, uint64 size) = 0;
        virtual const char* GetName() const = 0;
    };

    class HeapArena : public MemoryArena
    {
    public:
        HeapArena() = default;
        HeapArena(const char* name) : name(name) {}
        ~HeapArena() = default;
        HeapArena(const HeapArena& rhs) = delete;
        HeapArena& operator=(const HeapArena& rhs) = delete;
        void* Alloc(uint64 size, uint64 alignment = alignof(std::max_align_t)) override
        {
            return _aligned_malloc(size, alignment);
        }
        void Free(void* ptr, uint64 size) override
        {
            _aligned_free(ptr);
        } 
        char const* GetName() const override
        {
            return name;
        }
    private:
        const char* name = nullptr;
    };

    class LinearArena : public MemoryArena
    {
    public:
        LinearArena(const char* name, uint64 size)
            : name(name), size(size)
        {
            begin = malloc(size);
        }
        ~LinearArena()
        {
            free(begin);
        }
        LinearArena(const LinearArena& rhs) = delete;
        LinearArena& operator=(const LinearArena& rhs) = delete;
        void* Alloc(uint64 size, uint64 alignment)
        {
            void* const p = Align(GetCurrent(), alignment);
            void* const c = Add(p, size);
            bool success = (c <= Add(begin, this->size));
            if (success)
            {
                SetCurrent(c);
            }
            return success ? p : nullptr;
        }
        void Free(void* ptr, uint64 size)
        {

        } 
        void Reset() 
        {
            used = 0;
        }
        uint64 Allocated() const 
        {
            return used;
        }
        uint64 Available() const
        {
            return size - used;
        }  
        char const* GetName() const override
        {
            return name;
        }
    private:
        inline void* Add(void* ptr, uint64 offset)
        {
            return (void*)(uint64(ptr) + offset);
        }

        inline void* Align(void* ptr, uint64 alignment)
        {
            ASSERT(alignment && !(alignment & alignment - 1));
            return (void*)((uint64(ptr) + alignment - 1) & ~(alignment - 1));
        }
        void* GetCurrent()
        {
            return Add(begin, used);
        }
        void SetCurrent(void* ptr)
        { 
            used = uint64(ptr) - uint64(begin);
        }
    private:
        const char* name = nullptr;
        void* begin = nullptr;
        uint64 size = 0;
        uint64 used = 0;
    };

    extern LinearArena* GArena;

    void* ArenaRealloc(MemoryArena* arena, void* ptr, uint64 oldSize, uint64 newSize, uint64 alignment, const char* file, uint32 line);
}