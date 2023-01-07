#pragma once

#include "ECS/ECSCommon.h"
#include "ECS/Entity.h"

namespace HE
{
	struct AudioListenerComponent
	{
		uint64 id;
		AudioListenerComponent() = default;
		AudioListenerComponent(const AudioListenerComponent& other) = default;
	};
}
