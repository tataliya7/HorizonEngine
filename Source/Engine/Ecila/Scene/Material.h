#pragma once

#include "Ray.h"

enum class MaterialType
{ 
    DIFFUSE,
    DIFFUSE_AND_GLOSSY, 
    REFLECTION_AND_REFRACTION,
    REFLECTION,
    EMISSION
};

struct MaterialData
{
    Vector3 emission = { 0, 0, 0 };
    Real IOR;
    Vector3 Kd;
    Vector3 Ks;
    Real specularExponent;
    Real fuzz;
};

class Material
{
public:

    using SharedPtr = std::shared_ptr<Material>;

    static SharedPtr create(MaterialType type, const MaterialData& data);

    MaterialType getType() { return mType; }

    bool isLuminous() { return mData.emission.norm() > 0; }

    const MaterialData& getData() { return mData; }
    
    Vector3 getEmission() { return mData.emission; }

    Vector3 BRDF(const Vector3& wi, const Vector3& wo, const Vector3& N);

    Vector3 sample(const Vector3& wi, const Vector3& N);

    Real pdf(const Vector3& wi, const Vector3& wo, const Vector3& N);

protected:

    Material() = default;
    Material(MaterialType type, const MaterialData& data);
    MaterialType mType;
    MaterialData mData;
    //Texture tex;
};
