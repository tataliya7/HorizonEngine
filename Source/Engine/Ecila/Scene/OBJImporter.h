#pragma once

namespace Ecila
{
	class Scene;

	struct OBJImportSettings
	{

	};

	bool ImportOBJ(const char* filename, const OBJImportSettings& settings, Scene* scene);
}