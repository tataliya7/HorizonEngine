module;

#include <vector>

export module HorizonEngine.Animation;

import HorizonEngine.Core;

export namespace HE
{
	enum class AnimationInterpolationType
	{
		Linear,
		Step,
		CatmullRomSpline,
		CubicSpline,
	};

	struct AnimationTranslationTrack
	{
		std::vector<float> times;
		std::vector<Vector3> translations;
		AnimationInterpolationType interpolation;
	};

	struct AnimationRotationTrack
	{
		std::vector<float> times;
		std::vector<Quaternion> rotations;
		AnimationInterpolationType interpolation;
	};

	struct AnimationScaleTrack
	{
		std::vector<float> times;
		std::vector<Vector3> scales;
		AnimationInterpolationType interpolation;
	};

	struct AnimationChannel
	{
		int32 translationTrackIndex;
		int32 rotaionTrackIndex;
		int32 scaleTrackIndex;
	};

	class AnimationSequence
	{
	public:
		AnimationSequence();
		~AnimationSequence();
	private:
		float timeLength;
        /** The number of animation channels. Each channel affects a single node. */
        uint32 numChannels;
		std::vector<AnimationChannel> channels;
		std::vector<AnimationTranslationTrack> translationTracks;
		std::vector<AnimationRotationTrack> rotationTracks;
		std::vector<AnimationScaleTrack> scaleTracks;
	};
}
