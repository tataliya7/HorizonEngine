#include "BVH.h"
#include "Surface.h"

namespace Ecila
{
    AABB BVHNode::GetBounds() const
    {
        return bounds;
    }

    inline bool box_compare(const Surface& a, const Surface& b, int axis)
    {
        AABB aabb1 = a.GetBounds();
        AABB aabb2 = b.GetBounds();
        return aabb1.minPoint[axis] < aabb2.minPoint[axis];
    }

    bool box_x_compare(const Surface& a, const Surface& b)
    {
        return box_compare(a, b, 0);
    }

    bool box_y_compare(const Surface& a, const Surface& b)
    {
        return box_compare(a, b, 1);
    }

    bool box_z_compare(const Surface& a, const Surface& b)
    {
        return box_compare(a, b, 2);
    }

    BVHNode* BVH::BuildNode(std::vector<Surface>& surfaces, uint32 begin, uint32 end)
    {
        if (end <= begin)
        {
            return;
        }

        int axis = (int)(2.0f * Math::Random());
        auto comparator = (axis == 0) ? box_x_compare
            : (axis == 1) ? box_y_compare
            : box_z_compare;

        uint32 numSurfaces = end - begin;

        BVHNode* node = new BVHNode();
        node->numSurfaces = numSurfaces;
        node->firstSurface = begin;
        node->leftNode = nullptr;
        node->rightNode = nullptr;

        if (numSurfaces > 1)
        {
            std::sort(surfaces.begin() + begin, surfaces.begin() + end, comparator);
            auto mid = begin + numSurfaces / 2;
            node->leftNode = BuildNode(surfaces, begin, mid);
            node->rightNode = BuildNode(surfaces, mid, end);
        }

        if (numSurfaces == 1)
        {
            node->bounds = surfaces[begin].GetBounds();
        }
        else
        {
            AABB leftNodeBounds = node->leftNode->GetBounds();
            AABB rightNodeBounds = node->rightNode->GetBounds();
            node->bounds = AABB::Union(leftNodeBounds, rightNodeBounds);
        }
        return node;
    }

    void BVH::Build(std::vector<Surface>& surfaces)
    {
        root = BuildNode(surfaces, 0, (uint32)surfaces.size());
    }
}
