#pragma once

#include "EcilaCommon.h"
#include "EcilaMath.h"

namespace Ecila
{
    struct Transform
    {
        Transform() = default;
        Transform(const Vector3& p, const Vector3& r, const Vector3& s)
            : position(p), rotation(r), scale(s)
        {
            negative_determinant = s.x * s.y * s.z < 0.0;

            rotation_matrix = glm::rotate(r.z, Vector3(0.0f, 0.0f, 1.0f)) *
                glm::rotate(r.y, Vector3(0.0f, 1.0f, 0.0f)) *
                glm::rotate(r.x, Vector3(1.0f, 0.0f, 0.0f));

            matrix = glm::translate(glm::mat4(1.0f), p) *
                rotation_matrix *
                glm::scale(glm::mat4(1.0f), s);
        }
        Vector3 transformNormal(const Vector3& n) const
        {
            return rotation_matrix * glm::dvec4(glm::normalize(n / scale), 1.0);
        }
        glm::mat4 matrix, rotation_matrix;
        Vector3 position, scale, rotation;
        bool negative_determinant;
    };


    class Material
    {
    public:
        Material()
        {
            roughness = 0.0;
            specular_roughness = 0.0;
            ior = -1.0;
            complex_ior = nullptr;
            transparency = 0.0;
            perfect_mirror = false;
            reflectance = glm::dvec3(1.0);
            specular_reflectance = glm::dvec3(1.0);
            emittance = glm::dvec3(0.0);
            transmittance = glm::dvec3(1.0);

            computeProperties();
        }

        glm::dvec3 diffuseReflection(const glm::dvec3& i, const glm::dvec3& o, double& PDF) const;
        glm::dvec3 specularReflection(const glm::dvec3& wi, const glm::dvec3& wo, double& PDF) const;
        glm::dvec3 specularTransmission(const glm::dvec3& wi, const glm::dvec3& wo, double n1,
            double n2, double& PDF, bool inside, bool flux) const;

        glm::dvec3 visibleMicrofacet(double u, double v, const glm::dvec3& o) const;

        void computeProperties();

        glm::dvec3 reflectance, specular_reflectance, transmittance, emittance;
        double roughness, specular_roughness, ior, transparency;
        std::shared_ptr<ComplexIOR> complex_ior;

        bool rough, rough_specular, opaque, emissive, dirac_delta;

        // Represents ior = infinity -> fresnel factor = 1.0 -> all rays specularly reflected
        bool perfect_mirror;

    private:
        glm::dvec3 lambertian() const;
        glm::dvec3 OrenNayar(const glm::dvec3& wi, const glm::dvec3& wo) const;

        // Pre-computed Oren-Nayar variables.
        double A, B;

        // Specular roughness
        glm::dvec2 a;
    };

    class Mesh
    {
    public:
        Mesh() = default;
        std::vector<Vector3> vertices;
        std::vector<Vector3> normals;
        std::vector<Vector2> texCoords;
        std::vector<uint32> indices;

        Transform transform;
        Material* material;
    };
}
