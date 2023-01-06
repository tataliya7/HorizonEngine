#pragma once

#include "LinkedHorizonCommon.h"

namespace Ecila
{
	class Medium
	{
	public:

		Medium();

		virtual ~Medium() = default;

        bool operator=(const Medium other) const
        {
            return;
        }

        bool IsEmissive() const;

	private:

	};

    struct MediumInterface
    {
        MediumInterface() = default;
        MediumInterface(Medium m) : insideMedium(m), outsideMedium(m) {}
        MediumInterface(Medium im, Medium om) : insideMedium(im), outsideMedium(om) {}

        bool IsMediumTransition() const { return insideMedium != outsideMedium; }

        Medium insideMedium, outsideMedium;
    };
}
