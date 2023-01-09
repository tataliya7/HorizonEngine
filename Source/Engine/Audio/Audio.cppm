module;

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include <string>

export module HorizonEngine.Audio;

import HorizonEngine.Core;

export namespace HE
{
    namespace AudioEngine
    {
        ma_engine engine;

        bool Init()
        {
            ma_result result = ma_engine_init(NULL, &engine);
            if (result != MA_SUCCESS)
            {
                printf("Failed to initialize audio engine.");
                return false;
            }
            return true;
        }

        void Exit()
        {
            ma_engine_uninit(&engine);
        }

        void PlaySoundEX(const char* filename)
        {
            ma_engine_play_sound(&engine, filename, NULL);
        }

        void StopAll()
        {

        }

        void PauseAll()
        {

        }
        
        void ResumeAll()
        {

        }
    };

    struct AudioSourceComponent
    {
        std::string audio;

        float volumeMultiplier = 1.0f;

        AudioSourceComponent() = default;
        AudioSourceComponent(const AudioSourceComponent& other) = default;

        void Play()
        {
            AudioEngine::PlaySoundEX(audio.c_str());
        }
    };
}
