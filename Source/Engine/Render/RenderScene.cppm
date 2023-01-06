module;

#include <vector>
#include <ECS/ECS.h>

#include <optick.h>

export module HorizonEngine.Render.Scene;

import HorizonEngine.Core;
import HorizonEngine.Render.Core;
import HorizonEngine.Render.RenderGraph;
import HorizonEngine.Render.RenderPipeline;
import HorizonEngine.SceneManagement;

export namespace HE
{
	struct RenderBackend;
	
	struct Camera
	{
		float fieldOfView;
		float aspectRatio;
		float zNear;
		float zFar;
		Vector3 position;
		Vector3 euler;
		Matrix4x4 viewMatrix;
		Matrix4x4 invViewMatrix;
		Matrix4x4 projectionMatrix;
		Matrix4x4 invProjectionMatrix;
	};

#define MAX_NUM_SHADOW_MAP_CASCADES 4

	class LightRenderProxy
	{
	public:
		LightRenderProxy(const DirectionalLightComponent* lightComponent)
			: component(lightComponent) {}

		const DirectionalLightComponent* component;

		Vector3 direction;

		bool rayTracingShadows = false;
		uint32 numDynamicShadowCascades = 4;
		uint32 shadowMapSize = 4096;
		float cascadeSplitLambda = 0.95f;
		Matrix4x4 viewProjectionMatrix[MAX_NUM_SHADOW_MAP_CASCADES];
		float splitDepth[MAX_NUM_SHADOW_MAP_CASCADES];

		void UpdateShadowCascades(const Camera& camera)
		{
			float cascadeSplits[MAX_NUM_SHADOW_MAP_CASCADES];

			float nearClip = camera.zNear;
			float farClip = camera.zFar;

			//float nearClip = camera.zFar;
			//float farClip = camera.zNear;

			float clipRange = farClip - nearClip;

			float minZ = nearClip;
			float maxZ = nearClip + clipRange;

			float range = maxZ - minZ;
			float ratio = maxZ / minZ;

			// Calculate split depths based on view camera frustum
			// Based on method presented in https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch10.html
			for (uint32_t i = 0; i < MAX_NUM_SHADOW_MAP_CASCADES; i++)
			{
				float p = (i + 1) / static_cast<float>(MAX_NUM_SHADOW_MAP_CASCADES);
				float log = minZ * std::pow(ratio, p);
				float uniform = minZ + range * p;
				float d = cascadeSplitLambda * (log - uniform) + uniform;
				cascadeSplits[i] = (d - nearClip) / clipRange;
			}

			// Calculate orthographic projection matrix for each cascade
			float lastSplitDist = 0.0;
			for (uint32_t cascadeIndex = 0; cascadeIndex < MAX_NUM_SHADOW_MAP_CASCADES; cascadeIndex++)
			{
				float splitDist = cascadeSplits[cascadeIndex];

				glm::vec3 frustumCorners[8] = {
					glm::vec3(-1.0f,  1.0f,  1.0f),
					glm::vec3( 1.0f,  1.0f,  1.0f),
					glm::vec3( 1.0f, -1.0f,  1.0f),
					glm::vec3(-1.0f, -1.0f,  1.0f),
					glm::vec3(-1.0f,  1.0f,  0.0f),
					glm::vec3( 1.0f,  1.0f,  0.0f),
					glm::vec3( 1.0f, -1.0f,  0.0f),
					glm::vec3(-1.0f, -1.0f,  0.0f),
				};

				// Project frustum corners into world space
				glm::mat4 invCam = camera.invViewMatrix * camera.invProjectionMatrix;
				for (uint32_t i = 0; i < 8; i++)
				{
					glm::vec4 invCorner = invCam * glm::vec4(frustumCorners[i], 1.0f);
					frustumCorners[i] = invCorner / invCorner.w;
				}

				for (uint32_t i = 0; i < 4; i++) {
					glm::vec3 dist = frustumCorners[i + 4] - frustumCorners[i];
					frustumCorners[i + 4] = frustumCorners[i] + (dist * splitDist);
					frustumCorners[i] = frustumCorners[i] + (dist * lastSplitDist);
				}

				// Get frustum center
				glm::vec3 frustumCenter = glm::vec3(0.0f);
				for (uint32_t i = 0; i < 8; i++)
				{
					frustumCenter += frustumCorners[i];
				}
				frustumCenter /= 8.0f;

				float radius = 0.0f;
				for (uint32_t i = 0; i < 8; i++) 
				{
					float distance = glm::length(frustumCorners[i] - frustumCenter);
					radius = glm::max(radius, distance);
				}
				radius = std::ceil(radius * 16.0f) / 16.0f;

				glm::vec3 maxExtents = glm::vec3(radius);
				glm::vec3 minExtents = -maxExtents;

				glm::vec3 lightDir = direction;

				glm::mat viewMatrix = glm::lookAt(frustumCenter - lightDir * -minExtents.z, frustumCenter, glm::vec3(0.0f, 1.0f, 0.0f));
				//glm::mat projectionMatrix = glm::ortho(minExtents.x, maxExtents.x, minExtents.y, maxExtents.y, 0.0f, maxExtents.z - minExtents.z);
				glm::mat projectionMatrix = glm::ortho(minExtents.x, maxExtents.x, minExtents.y, maxExtents.y, maxExtents.z - minExtents.z, 0.0f);

				viewProjectionMatrix[cascadeIndex] = projectionMatrix * viewMatrix;
				splitDepth[cascadeIndex] = (nearClip + splitDist * clipRange) * -1.0f;

				lastSplitDist = cascadeSplits[cascadeIndex];
			}
		}
	};

	class SkyLightRenderProxy
	{
	public:
		SkyLightRenderProxy(const SkyLightComponent* component)
			: component(component) {}

		const SkyLightComponent* component;

		RenderBackendTextureHandle environmentMap;
		RenderBackendTextureHandle irradianceEnvironmentMap;
		RenderBackendTextureHandle filteredEnvironmentMap;
	};

	struct PBRMaterialShaderParameters
	{
		int32 flags;
		Vector4 baseColor;
		float metallic;
		float roughness;
		Vector4 emission;
		uint32 baseColorMapIndex;
		uint32 normalMapIndex;
		uint32 metallicRoughnessMapIndex;
		uint32 emissiveMapIndex;
	};

	struct Renderable
	{
		Matrix4x4 localMatrix;
		uint32 firstVertex;
		uint32 firstIndex;
		uint32 numIndices;
		uint32 numVertices;
		uint32 vertexBufferIndex;
		uint32 indexBufferIndex;
		uint32 materialIndex;
		uint32 transformIndex;
	};

	class MeshRenderProxy
	{
	public:
		MeshRenderProxy(const StaticMeshComponent* component)
			: component(component) {}
		const StaticMeshComponent* component;

		std::vector<Renderable> renderables;

		Matrix4x4 worldMatrix;

	private:

	};

	class DaisyRenderer;

	class RenderScene
	{
	public:
		RenderScene();
		~RenderScene();

		Scene* scene;
		
		// TODO: Remove this
		void Setup(Scene* scene, DaisyRenderer* renderer);
		void Setup(Scene* scene, RenderBackend* renderBackend, MemoryArena* arena);

		void Update(float deltaTime);

		void UpdateSkyLights();

		RenderBackend* renderBackend;

		std::vector<MeshRenderProxy*> meshes;

		std::vector<Renderable> renderables;
		std::vector<RenderBackendBufferHandle> vertexBuffers[4];
		std::vector<RenderBackendBufferHandle> indexBuffers;
		std::vector<RenderBackendTextureHandle> textures;

		std::vector<PBRMaterialShaderParameters> materials;
		RenderBackendBufferHandle materialBuffer;

		std::vector<Matrix4x4> worldMatrices;
		std::vector<Matrix4x4> prevWorldMatrices;
		RenderBackendBufferHandle worldMatrixBuffer;
		RenderBackendBufferHandle prevWorldMatrixBuffer;

		RenderBackendBufferHandle worldMatrixBuffer1;
		RenderBackendRayTracingAccelerationStructureHandle bottomLevelAS;
		RenderBackendRayTracingAccelerationStructureHandle topLevelAS;

		void SetSkyLight(SkyLightRenderProxy* proxy);
		SkyLightRenderProxy* skyLight;

		LightRenderProxy* simpleDirectionalLight;
	};

	struct SceneView
	{
		SceneView() = default;
		RenderPipeline* renderPipeline;
		RenderScene* scene;
		Camera camera;
		uint32 targetWidth;
		uint32 targetHeight;
		RenderBackendTextureDesc targetDesc;
		RenderBackendTextureHandle target;
		RenderBackendTextureDesc captureTargetDescs[8];
		RenderBackendTextureHandle captureTargets[8];
	};

	void RenderSceneView(RenderContext* renderContext, SceneView* sceneView);

	void RenderSceneView(
		RenderContext* renderContext,
		SceneView* view)
	{
		OPTICK_EVENT();

		MemoryArena* arena = renderContext->arena;
		RenderBackend* renderBackend = renderContext->renderBackend;
		RenderPipeline* activePipeline = view->renderPipeline;
		uint32 deviceMask = ~0u;

		// UpdateCameraMatrices(view->camera, (float)(view->targetWidth / view->targetHeight));
		{
			RenderGraph renderGraph(arena);
        
			activePipeline->SetupRenderGraph(view, &renderGraph);
        
			renderContext->commandLists.clear();
			// HE_LOG_INFO(renderGraph.Graphviz());
			renderGraph.Execute(renderContext);
		}

		uint32 numCommandLists = (uint32)renderContext->commandLists.size();
		RenderCommandList** commandLists = renderContext->commandLists.data();
		RenderBackendSubmitRenderCommandLists(renderBackend, commandLists, numCommandLists);
	}
}