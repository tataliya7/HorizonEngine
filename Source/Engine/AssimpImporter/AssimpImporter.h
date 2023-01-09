#pragma once

import HorizonEngine.Core;

namespace HE
{
    struct OBJImportSettings
    {

    };

    extern bool ImportOBJ(const char* filename, OBJImportSettings settings);

    struct FBXImportSettings
    {

    };

    extern bool ImportFBX(const char* filename, FBXImportSettings settings);
    
    struct GLTF2ImportSettings
    {

    };

    extern bool ImportGLTF2(const char* filename, GLTF2ImportSettings settings);

    class AssimpImporter : public AssetImporter
    {
    public:
        void ImportAsset(const char* filename) override;
    };
}
