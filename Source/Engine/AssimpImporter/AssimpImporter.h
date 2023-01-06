#pragma once

import HorizonEngine.Core;
import HorizonEngine.Render;
import HorizonEngine.SceneManagement;

namespace HE
{
    class RenderScene;

    struct OBJImportSettings
    {

    };

    extern bool ImportOBJ(const char* filename, OBJImportSettings settings, RenderScene* scene);

    struct FBXImportSettings
    {

    };

    extern bool ImportFBX(const char* filename, FBXImportSettings settings, RenderScene* scene);
    
    struct GLTF2ImportSettings
    {

    };

    extern bool ImportGLTF2(const char* filename, GLTF2ImportSettings settings, RenderScene* scene);

    class AssimpImporter : public AssetImporter
    {
    public:
        void ImportAsset(const char* filename) override;
    };
}
