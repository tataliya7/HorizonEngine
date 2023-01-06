#pragma once

#define GLM_FORCE_CTOR_INIT
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#define M_PI 				  (3.1415926535897932f)	
#define M_INV_PI			  (0.3183098861837067f)
#define M_HALF_PI			  (1.5707963267948966f)
#define M_TWO_PI			  (6.2831853071795864f)
#define M_PI_SQUARED		  (9.8696044010893580f)
#define M_SQRT_PI	      	  (1.4142135623730950f)
#define SMALL_NUMBER		  (1.e-8f)
#define KINDA_SMALL_NUMBER    (1.e-4f)
#define BIG_NUMBER			  (3.4e+38f)
#define DELTA			      (0.00001f)
#define FLOAT_MAX			  (3.402823466e+38f)