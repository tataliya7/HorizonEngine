export module HorizonEngine.ParticleSystem;

import <queue>;
import HorizonEngine.Core;

export namespace HE
{
	struct Particle
	{
		Vector3 position;
		float age;
		Vector3 velocity;
		float lifetime;
		float id;
		float random;
	};

	enum class ParticleEmittionShape
	{
		Point = 0,
		Line = 1,
		Ring = 2,
		Cube = 3,
		Sphere = 4,
		Rectangle = 5,
		Cone = 6,
		Mesh = 7,
	};

	enum class ParticleMotionPrimitiveType
	{
		Point,
	};

	struct ParticlePrimitive
	{
		Vector3 position;
		float strength;
	};

	struct ParticleEmission
	{
		float emissionRate;
		float lifetime;
		float startSpeed;
		int initialParticles;
		int maxParticles;
		int shape;
		float radius;
		float thickness;
		float length;
		float angle;
	};

	struct ParticleMotion
	{
		float simulationSpeed;
		Vector3 gravity;
		Vector3 pointAttractor;
	};

	struct ParticleAppearance
	{
		Vector4 startColor;
		Vector4 endColor;
		float size;
		float brightness;
	};

	class ParticleSystem
	{
	public:

		ParticleSystem();
		~ParticleSystem();

		struct Allocation
		{
			float timeElapsed;
			uint32 numParticlesEmitted;
		};
		std::queue<Allocation> allocationHistory;

		bool looping;

		float duration;

		float timeElapsed;

	private:

		ParticleEmission emission;
		ParticleMotion motion;
		ParticleAppearance appearance;
	};

	struct ParticleSystemComponent
	{

	};

	struct ParticleSystemRenderProxy
	{

	};
}
