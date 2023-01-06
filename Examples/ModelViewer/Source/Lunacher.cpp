#pragma once

#include "ModelViewer.h"

#define HE_JOB_SYSTEM_NUM_FIBIERS 128
#define HE_JOB_SYSTEM_FIBER_STACK_SIZE (HE_JOB_SYSTEM_NUM_FIBIERS * 1024)

int main(int argc, char** argv)
{
	HE::LogSystemInit();
	HE::JobSystemInit(HE::GetNumberOfProcessors(), HE_JOB_SYSTEM_NUM_FIBIERS, HE_JOB_SYSTEM_FIBER_STACK_SIZE);
	int exit = ModelViewerMain();
	HE::JobSystemExit();
	HE::LogSystemExit();
	return exit;
}