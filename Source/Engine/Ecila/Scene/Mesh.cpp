#include "Mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

Mesh::SharedPtr Mesh::loadFromFile(String filename, Material::SharedPtr material)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str());
    if (!err.empty()) 
    {
        std::cerr << "ERR: " << err << std::endl;
    }
    if (!ret) 
    {
        std::cerr << "Failed to load/parse .obj." << std::endl;
    }
    assert(shapes.size() == 1);

    auto meshData = shapes[0].mesh;

	SharedPtr pMesh = SharedPtr(new Mesh());

    Vector3 min_vert = { std::numeric_limits<Real>::infinity(),
                         std::numeric_limits<Real>::infinity(),
                         std::numeric_limits<Real>::infinity() };
    Vector3 max_vert = { -std::numeric_limits<Real>::infinity(),
                         -std::numeric_limits<Real>::infinity(),
                         -std::numeric_limits<Real>::infinity() };

    size_t index_offset = 0;

    for (size_t i = 0; i < attrib.vertices.size() / 3; i++)
    {
        Vector3 vert;
        for (size_t v = 0; v < 3; v++)
        {
            vert[v] = attrib.vertices[i * 3 + v];
        }
        pMesh->mVertices.push_back(vert);

        min_vert = Vector3(std::min(min_vert[0], vert[0]),
            std::min(min_vert[1], vert[1]),
            std::min(min_vert[2], vert[2]));
        max_vert = Vector3(std::max(max_vert[0], vert[0]),
            std::max(max_vert[1], vert[1]),
            std::max(max_vert[2], vert[2]));
    }
    
    pMesh->mArea = 0;
	// For each face
    for (size_t i = 0; i < meshData.num_face_vertices.size(); i++)
    {
		size_t fnum = meshData.num_face_vertices[i];
        assert(fnum == 3);
		tinyobj::index_t idx;
        pMesh->mIndices.resize(meshData.indices.size());
		for (size_t v = 0; v < fnum; v++)
		{
			idx = meshData.indices[index_offset + v];
            pMesh->mIndices[index_offset + v] = idx.vertex_index;
		}
        pMesh->mTriangles.emplace_back(pMesh->mVertices[pMesh->mIndices[index_offset]],
            pMesh->mVertices[pMesh->mIndices[index_offset + 1]],
            pMesh->mVertices[pMesh->mIndices[index_offset + 2]],
            material);
		index_offset += fnum;

        pMesh->mArea += pMesh->mTriangles[i].getArea();
    }
    pMesh->mBounds = AABB(min_vert, max_vert);

    std::vector<Hittable::SharedPtr> pHittables;
    for (auto& tri : pMesh->mTriangles)
    {
        pHittables.push_back((Hittable::SharedPtr)&tri);
    }
    pMesh->mpBVHNode = (BVHNode::SharedPtr)new BVHNode(pHittables);
    
    return pMesh;
}

bool Mesh::hit(const Ray& ray, Real tMin, Real tMax, HitRecord& record) const
{
    return mpBVHNode->hit(ray, tMin, tMax, record);
}

bool Mesh::getAABB(AABB& outputAABB) const
{
    outputAABB = mBounds;
    return true;
}
