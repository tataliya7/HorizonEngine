#pragma once

#define POST_PROCESSING_THREAD_GROUP_SIZE 8

struct PostProcessingSettings
{
	float exposureCompensation;
	float autoExposureMinExposureValue;
	float autoExposureMaxExposureValue;
	float autoExposureSpeedUp;
	float autoExposureSpeedDown;
};