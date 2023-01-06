#include "Camera.h"
#include "Sampling.h"

namespace Ecila
{
    static_assert(sizeof(CameraParameters) % (4 * sizeof(float)) == 0);

    SharedPtr<PerspectiveCamera> PerspectiveCamera::Create()
    {
        return SharedPtr<PerspectiveCamera>(new PerspectiveCamera());
    }

    void PerspectiveCamera::Update()
    {
        if (IsDirty())
        {
            mParameters.targetPoint = mParameters.posW - glm::normalize(mParameters.W);
            mParameters.view = glm::lookAt(mParameters.posW, mParameters.targetPoint, mParameters.V);
            mParameters.proj = glm::perspective(mParameters.yFov / 2.0f, mParameters.aspectRatio, mParameters.nearZ, mParameters.farZ);
            mParameters.viewProj = mParameters.proj * mParameters.view;
            mParameters.invViewProj = glm::inverse(mParameters.viewProj);
            SetDirty(false);
        }
    }

    Ray PerspectiveCamera::GenerateRay(const CameraSampleInfo& info) const
    {
        Vector3 filmPos = Vector3(sample.pFilm.x, sample.pFilm.y, mParameters.focalLength);
        Vector3 direction = -filmPos;

        // With the perspective projection, all rays originate from the origin in camera space.
        Ray ray(Vector3(0, 0, 0), glm::normalize(samplePos), SampleTime(sample.time), &mMedium);

        if (mParameters.aperture > 0)
        {
            Vector2 newOrigin = mParameters.aperture / 2.0f * SampleUniformDiskConcentric(info.pos);

            /**
             * @note We call the distance between the projection point and the plane where everything is in perfect focus the focus distance. 
             * Be aware that the focus distance is not the same as the focal length.
             * The focal length is the distance between the projection point and the image plane.
             * In a physical camera, the focus distance is controlled by the distance between the lens and the film/sensor.
             * @see [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
             */
            float t = mParameters.focusDistance / ray.mDirection.z;
            Vector3 pointOnTheFocalPlane = ray.At(t);

            ray.mOrigin = Vector3(newOrigin.x, newOrigin.y, 0);
            ray.mDirection = glm::normalize(pointOnTheFocalPlane - ray.mOrigin);
        }
        return CameraRay{ RenderFromCamera(ray) };
    }

    Ray PerspectiveCamera::GenerateRayDifferential(const CameraSampleInfo&& info) const
    {
        return Ray();
    }
}
