#pragma once

import HorizonEngine.Core;
import HorizonEngine.Audio;

namespace HE
{
    class AudioController : public Scriptable
    {
    public:

        AudioController() = default;
        virtual ~AudioController() = default;

        void OnCreate()
        {
            audioSource = TryGetComponent<AudioSourceComponent>();
            audioSource->Play();
        }

        void OnDestroy()
        {

        }

        void OnUpdate(float deltaTime)
        {

        }

    private:

        AudioSourceComponent* audioSource;
    };
}