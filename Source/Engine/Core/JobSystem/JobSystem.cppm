export module HorizonEngine.Core.JobSystem;

import HorizonEngine.Core.Types;

export namespace HE
{
    enum
    {
        JOB_SYSTEM_MAX_NUM_WORKER_THREADS = 128,
        JOB_SYSTEM_MAX_NUM_FIBERS = 256,
        JOB_SYSTEM_MAX_NUM_JOBS = 4096,
    };

    using JobSystemAtomicCounterHandle = uint32;

    using JobSystemJobFunc = void(*)(void*);

    struct JobSystemJobDecl
    {
        JobSystemJobFunc jobFunc;
        void* data;
    };

    void JobSystemInit(uint32 numWorkerThreads, uint32 numFibers, uint32 fiberStackSize);
    void JobSystemExit();
    JobSystemAtomicCounterHandle JobSystemRunJobs(JobSystemJobDecl* jobDecls, uint32 numJobs);
    void JobSystemWaitForCounter(JobSystemAtomicCounterHandle counterHandle, uint32 condition);
    void JobSystemWaitForCounterAndFree(JobSystemAtomicCounterHandle counterHandle, uint32 condition);
    void JobSystemWaitForCounterAndFreeWithoutFiber(JobSystemAtomicCounterHandle counterHandle);
}