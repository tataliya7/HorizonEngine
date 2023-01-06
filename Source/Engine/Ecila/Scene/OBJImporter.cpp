#include "OBJImporter.h"
#include "EcilaCommon.h"
#include "EcilaMath.h"
#include "Surface.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace Ecila
{
	bool ImportOBJ(const char* filename, const OBJImportSettings& settings, Scene* scene)
	{
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string err;

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename);
        if (!err.empty())
        {
            std::cout << "ERR: " << err << std::endl;
            return false;
        }
        if (!ret)
        {
            std::cout << "Failed to load/parse .obj." << std::endl;
            return false;
        }
        assert(shapes.size() == 1);

        auto mesh = shapes[0].mesh;

        auto& surfaces = scene->surfaces;

        // For each triangle
        for (size_t i = 0; i < mesh.num_face_vertices.size(); i++)
        {
            size_t numVertices = mesh.num_face_vertices[i];
            assert(numVertices == 3);

            tinyobj::index_t index1 = mesh.indices[i * 3 + 0];
            tinyobj::index_t index2 = mesh.indices[i * 3 + 1];
            tinyobj::index_t index3 = mesh.indices[i * 3 + 2];

            Surface surface = Surface(
                attrib.vertices[index1],
                attrib.vertices[index2],
                attrib.vertices[index3],
                attrib.normals[index1],
                attrib.normals[index2],
                attrib.normals[index3]
                material);

            surfaces.emplace_back(surface);
        }
        
        return true;
	}
}