#include "Material.h"

Material::SharedPtr Material::create(MaterialType type, const MaterialData& data)
{
	return SharedPtr(new Material(type, data));
}


Material::Material(MaterialType type, const MaterialData& data)
	: mType(type)
	, mData(data)
{

}

Vector3 Material::BRDF(const Vector3& wi, const Vector3& wo, const Vector3& N) 
{
    switch (mType)
    {
    case MaterialType::DIFFUSE:
    {
        Real cosAlpha = N.dot(wo);
        if (cosAlpha > 0.0f) 
        {
            Vector3 diffuse = mData.Kd / PI;
            return diffuse;
        }
        else
            return Vector3(0, 0, 0);
        break;
    }
    }
}

Vector3 toWorld(const Vector3& a, const Vector3& N)
{
    Vector3 B, C;
    if (std::fabs(N.x()) > std::fabs(N.y()))
    {
        Real invLen = 1.0f / std::sqrt(N.x() * N.x() + N.z() * N.z());
        C = Vector3(N.z() * invLen, 0.0f, -N.x() * invLen);
    }
    else {
        Real invLen = 1.0f / std::sqrt(N.y() * N.y() + N.z() * N.z());
        C = Vector3(0.0f, N.z() * invLen, -N.y() * invLen);
    }
    B = C.cross(N);
    return a.x() * B + a.y() * C + a.z() * N;
}

Vector3 Material::sample(const Vector3& wi, const Vector3& N)
{
    switch (mType)
    {
    case MaterialType::DIFFUSE:
    {
        // Uniform sample on the hemisphere
        Real x_1 = random(), x_2 = random();
        Real z = std::fabs(1.0f - 2.0f * x_1);
        Real r = std::sqrt(1.0f - z * z);
        Real phi = 2 * PI * x_2;
        Vector3 localRay(r * std::cos(phi), r * std::sin(phi), z);
        return toWorld(localRay, N);

        break;
    }
    }
}

Real Material::pdf(const Vector3& wi, const Vector3& wo, const Vector3& N)
{
    switch (mType)
    {
    case MaterialType::DIFFUSE:
    {
        if (wo.dot(N) > 0.0f)
            return 0.5f / PI;
        else
            return 0.0f;
        break;
    }
    }
}
