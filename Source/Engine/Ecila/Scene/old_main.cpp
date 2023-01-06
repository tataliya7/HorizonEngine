#include "Ecila.h"
#include "OBJImporter.h"

using namespace Ecila;

int main()
{
    uint32 width = 1104;
    uint32 height = 400;

    Camera* camera = new Camera();
    camera->focalLength = 46.0f / 1000.0f;
    camera->sensorWidth = 35.0f / 1000.0f;
    camera->position = { 0.0f, 0.015f, 0.42f };
    
    Vector3 targetPoint = { 0.0f, 0.048f, 0.0f };
    camera->LookAt(targetPoint);

    camera->focusDistance = glm::distance(camera->position, targetPoint);

    Film film = Film(width, height);
    camera->film = &film;

    const uint32 spp = 16;

    Framebuffer* framebuffer = new Framebuffer(width, height, Vector3(0, 0, 0));

    Scene* scene = new Scene();
    OBJImportSettings settings;
    
    ImportOBJ("D:/3DModels/shell/shell.obj", settings, scene);
    ImportOBJ("D:/3DModels/shell/shell.obj", settings, scene);
    ImportOBJ("D:/3DModels/shell/floor.obj", settings, scene);

    scene->meshes[0]->transform = Transform(
        Vector3(-0.0592f, 0.03848f, 0.0f),
        Vector3(-184.26f, 185.79f, 11.01f),
        Vector3(0.001f)
    );
    scene->meshes[0]->material = ;
    
    scene->meshes[1]->transform = Transform(
        Vector3(0.059f, 0.03848f, 0.0f),
        Vector3(-175.74f, 185.79f, 168.99f),
        Vector3(0.001f)
    );
    scene->meshes[1]->material = ;
    
    scene->meshes[2]->transform = Transform(
        Vector3(0.0f),
        Vector3(0.0f),
        Vector3(1.0f)
    );
    scene->meshes[2]->material = ;

    Light light1 = {
        .rectangle = {
            Vector3(-0.359309f, 0.449693f, -0.010809f),
            Vector3(-0.196537f, 0.449693f, 0.338256f),
            Vector3(-0.196537f, 0.000849009f, 0.338256f),
            Vector3(-0.359309f, 0.000848979f, -0.010809f),
        },
    };
    Light light2 = {
        .rectangle = {
            Vector3(0.320673f, 0.027337f, 0.228975f),
            Vector3(0.320673f, 0.476182f, 0.228975f),
            Vector3(0.325221f, 0.476182f, -0.136419f),
            Vector3(0.325221f, 0.027337f, -0.136419f),
        },
    };
    Light light3 = {
        .rectangle = {
            Vector3(0.230128f, 0.50385f, 0.267372f),
            Vector3(-0.230128f, 0.50385f, 0.267372f),
            Vector3(-0.230128f, 0.50385f, -0.192885f),
            Vector3(0.230128f, 0.50385f, -0.192885f),
        },
    };

    scene->lights.push_back(light1);
    scene->lights.push_back(light2);
    scene->lights.push_back(light3);

    scene->Update();
    scene->BuildBVH();

    EcilaRenderEngine* renderEngine = new EcilaRenderEngine();
    renderEngine->SetSPP(spp);
    renderEngine->RenderOneFrame(camera, scene, framebuffer);

    return 0;
}
