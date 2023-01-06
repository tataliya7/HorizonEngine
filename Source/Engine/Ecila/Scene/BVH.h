#pragma once

#include "Surface.h"
#include "Bounds.h"
#include "Intersection.h"

namespace Ecila
{
    class RayTracingAccelerationStucture
    {

    };

    class BVH
    {
    public:
        void Build(std::vector<Surface>& surfaces);
        bool IsEmpty() const
        {
            return root == nullptr;
        }
        BVHNode* GetRoot()
        {
            return root;
        }
    private:
        friend class Scene;
        BVHNode* BuildNode(std::vector<Surface>& surfaces, uint32 begin, uint32 end);
        BVHNode* root;
        std::vector<BVHNode*> nodes;
    };

    class BVHNode
    {
    public:
        BVHNode() = default;
        ~BVHNode() = default;
        bool IsBottomLevelAS() const
        {
            return leftNode == nullptr && rightNode == nullptr;
        }
        AABB GetBounds() const
        {
            return bounds;
        }
        uint32 GetFirstSurface() const
        {
            return firstSurface;
        }
        uint32 GetSurfaceCount() const
        {
            return numSurfaces;
        }
    private:
        friend class BVH;
        uint32 numSurfaces;
        uint32 firstSurface;
        BVHNode* leftNode;
        BVHNode* rightNode;
        AABB bounds;
    };
}