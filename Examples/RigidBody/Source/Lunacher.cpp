#pragma once

#include "HorizonExample.h"

#define HE_JOB_SYSTEM_NUM_FIBIERS 128
#define HE_JOB_SYSTEM_FIBER_STACK_SIZE (HE_JOB_SYSTEM_NUM_FIBIERS * 1024)

HORIZON_EXAMPLE_MAIN()

int main(int argc, char** argv)
{
	HE::LogSystemInit();
	HE::JobSystemInit(HE::GetNumberOfProcessors(), HE_JOB_SYSTEM_NUM_FIBIERS, HE_JOB_SYSTEM_FIBER_STACK_SIZE);
	int exit = HorizonExampleMain();
	HE::JobSystemExit();
	HE::LogSystemExit();
	return exit;
}