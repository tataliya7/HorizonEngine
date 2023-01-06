#pragma once

#include "ECS/ECSCommon.h"
#include "ECS/Entity.h"

#define TAG_COMPONENT_MAX_NUM_TAGS 32

namespace HE
{
    struct Tag
    {
        EntityHandle entity;
        std::pmr::string tag;
        union
        {
            uint32 next_by_entity;
            uint32 next_free;
        };
        uint32 prev_by_entity;
        uint32 next_by_tag;
        uint32 prev_by_tag;
    };

    struct TagComponent
    {
	    uint32 fistTagIndex;
    };

    void AddTag();
    void RemoveTag();
    bool HasTag();
    void FindAllEntities();
    EntityHandle FindFirstEntity();
}
