#pragma once

#include "EcilaCommon.h"
#include "EcilaMath.h"
#include "Surface.h"
#include "Intersection.h"

namespace Ecila
{
    struct LightSource
    {

    };

    class Mesh;
    class Light;
    class BVH;
    class BVHNode;

    class Scene
    {
    public:

        Scene() = default;

        void Update();

        void BuildBVH();

        HitRecord TraversalAccelerations(const Ray& ray, float tMin, float tMax) const;

        std::vector<Mesh*> meshes;
        std::vector<Light*> lights;

    private:
        
        void BuildBVHNode(BVHNode* node, uint32 begin, uint32 end);

        BVH* bvh;

        std::vector<Surface> surfaces;
        std::vector<LightSource> lightSources;
    };
}
