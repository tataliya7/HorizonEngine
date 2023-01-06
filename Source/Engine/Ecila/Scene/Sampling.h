#pragma once

#include "LinkedHorizonMath.h"

namespace Ecila
{
    inline Vector2 SampleUniformDiskConcentric(const Vector2& u)
    {
        Vector2 uOffset = 2.0f * u - Vector2(1, 1);
        if (uOffset.x == 0 && uOffset.y == 0)
        {
            return Vector2(0, 0);
        }

        float theta, r;
        if (std::abs(uOffset.x) > std::abs(uOffset.y))
        {
            r = uOffset.x;
            theta = PI / 4 * (uOffset.y / uOffset.x);
        }
        else {
            r = uOffset.y;
            theta = PI / 2 - PI / 4 * (uOffset.x / uOffset.y);
        }
        return r * Vector2(std::cos(theta), std::sin(theta));
    }
}
