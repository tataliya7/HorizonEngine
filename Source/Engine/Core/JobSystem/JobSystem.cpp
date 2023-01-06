module;

#include "CoreCommon.h"

#include <windows.h>
#include <MPMCQueue.h>
#include <optick.h>

module HorizonEngine.Core.JobSystem;

import HorizonEngine.Core.Logging;

namespace HE
{
    /**
     * A bounded multi-producer multi-consumer concurrent queue written in C++11.
     * Source code: https://github.com/rigtorp/MPMCQueue.
     */
    template<typename T>
    using MPMCQueue = rigtorp::mpmc::Queue<T>;

    struct Job
    {
        JobSystemJobDecl decl;
        JobSystemAtomicCounterHandle counterHandle;
    };

    struct WorkerThread
    {
        uint64 handle;
        uint32 id;
    };

    struct Semaphore
    {
        uint64 handle;
    };

    struct Fiber;
    struct SleepingFiber
    {
        uint32 condition;
        JobSystemAtomicCounterHandle counterHandle;
        Fiber* fiber;
    };

    struct Fiber
    {
        uint64 handle;
        uint32 index;
        SleepingFiber sleepingFiberToSchedule;
    };

    typedef void ThreadEntryFunction(void* userData);
    typedef void FiberEntryFunction(void* userData);

    struct FiberData
    {
        FiberEntryFunction* fiberEntry;
        void* userData;
    };

    struct ThreadData
    {
        ThreadEntryFunction* threadEntry;
        void* userData;
    };

    struct WorkerThreadUserData
    {
        uint32 workerThreadIndex;
        std::atomic<uint32>* bootAtomic;
    };

    struct AtomicCounter
    {
        uint32 index;
        std::atomic<uint32> atomic;
    };

    std::atomic<bool> gInitialized;
    uint32 gWorkerThreadCount;
    WorkerThread gWorkerThreads[JOB_SYSTEM_MAX_NUM_WORKER_THREADS];
    uint32 gWorkerThreadIDs[JOB_SYSTEM_MAX_NUM_WORKER_THREADS];
    uint32 gFiberCount;
    Fiber gFibers[JOB_SYSTEM_MAX_NUM_FIBERS];
    AtomicCounter gAtomicCounters[JOB_SYSTEM_MAX_NUM_JOBS];
    MPMCQueue<uint32> gFreeFiberQueue(JOB_SYSTEM_MAX_NUM_FIBERS);
    MPMCQueue<SleepingFiber> gSleepingFiberQueue(JOB_SYSTEM_MAX_NUM_FIBERS);
    MPMCQueue<Job> gJobQueue(JOB_SYSTEM_MAX_NUM_JOBS);
    MPMCQueue<uint32> gFreeCounterQueue(JOB_SYSTEM_MAX_NUM_JOBS);
    Semaphore gSemaphores[JOB_SYSTEM_MAX_NUM_WORKER_THREADS];
    std::map<uint32, uint32> gThreadIdSemaphoreLUT;
    std::atomic<uint32> gNextWorkerThreadIndex;
    ThreadData gThreadData[JOB_SYSTEM_MAX_NUM_WORKER_THREADS];
    WorkerThreadUserData gWorkerThreadUserData[JOB_SYSTEM_MAX_NUM_WORKER_THREADS];
    FiberData gFiberData[JOB_SYSTEM_MAX_NUM_FIBERS];

#if HE_PLATFORM_WINDOWS
    static uint32 GetNumberOfProcessors()
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return si.dwNumberOfProcessors;
    }

    static void SuspendCurrentThread(float seconds)
    {
        Sleep((DWORD)(seconds * 1000.0f + 0.5f));
    }

    static void YieldCPU()
    {
        YieldProcessor();
    }

    static uint32 GetCurrentThreadID()
    {
        return GetCurrentThreadId();
    }

    static void SwitchToAnotherFiber(uint64 handle)
    {
        SwitchToFiber((void*)handle);
    }

    static uint64 CreateSemaphoreEXT(uint32 initialCount)
    {
        uint64 handle = (uint64)CreateSemaphoreW(NULL, initialCount, INT_MAX, NULL);
        return handle;
    }

    static void SemaphoreAdd(uint64 semaphore, uint32 count)
    {
        ReleaseSemaphore((HANDLE)semaphore, count, NULL);
    }

    static void SemaphoreWait(uint64 semaphore)
    {
        WaitForSingleObject((HANDLE)semaphore, 0xFFFFFFFF);
    }

    static FiberData* GetCurrentFiberData()
    {
        ASSERT(IsThreadAFiber());
        return (FiberData*)GetFiberData();
    }

    static uint64 ConvertCurrentThreadToFiber(void* fiber)
    {
        return (uint64)ConvertThreadToFiberEx(fiber, FIBER_FLAG_FLOAT_SWITCH);
    }

    static bool ConvertCurrentFiberToThread()
    {
        return ConvertFiberToThread();
    }

    static DWORD WINAPI ThreadProc(LPVOID lpThreadParameter)
    {
        ThreadData* threadData = (ThreadData*)lpThreadParameter;
        threadData->threadEntry(threadData->userData);
        return 0;
    }

    static VOID WINAPI FiberProc(LPVOID lpFiberParameter)
    {
        FiberData* fiberData = (FiberData*)lpFiberParameter;
        fiberData->fiberEntry(fiberData->userData);
    }

    static uint64 CreateFiber(uint32 stackSize, FiberData* fiberData)
    {
        uint64 handle = (uint64)CreateFiberEx(stackSize, stackSize, FIBER_FLAG_FLOAT_SWITCH, FiberProc, fiberData);
        return handle;
    }

    static WorkerThread CreateWokerThread(uint32 stackSize, ThreadData* threadData, const wchar_t* description)
    {
        DWORD threadID;
        HANDLE handle = CreateThread(NULL, stackSize, ThreadProc, threadData, CREATE_SUSPENDED, &threadID);
        ASSERT(handle);

        if (description)
        {
            SetThreadDescription(handle, description);
        }

        ResumeThread(handle);

        WorkerThread thread;
        memcpy(&thread.handle, &handle, sizeof(handle));
        thread.id = threadID;

        return thread;
    }
#endif

    static uint32 LoadCounter(JobSystemAtomicCounterHandle handle)
    {
        ASSERT(handle);
        uint32 index = handle - 1;
        return gAtomicCounters[index].atomic.load(std::memory_order_acquire);
    }

    static void StoreCounter(JobSystemAtomicCounterHandle handle, uint32 value)
    {
        ASSERT(handle);
        uint32 index = handle - 1;
        gAtomicCounters[index].atomic.store(value);
    }

    static void FetchSubCounter(JobSystemAtomicCounterHandle handle)
    {
        ASSERT(handle);
        uint32 index = handle - 1;
        gAtomicCounters[index].atomic.fetch_sub(1);

    }

    static void FreeCounter(JobSystemAtomicCounterHandle handle)
    {
        ASSERT(handle);
        uint32 index = handle - 1;
        gFreeCounterQueue.push(gAtomicCounters[index].index);
    }

    static bool FindFreeCounter(uint32& outIndex)
    {
        return gFreeCounterQueue.try_pop(outIndex);
    }

    static bool FindFreeFiber(uint32& outIndex)
    {
        return gFreeFiberQueue.try_pop(outIndex);
    }

    static void FiberEntry(void* userData)
    {
        Fiber* currentFiber = (Fiber*)userData;

        while (!gInitialized)
        {
            YieldCPU();
        }

        SleepingFiber sleepingFiber;
        Job job;

        while (true)
        {
            if (currentFiber->sleepingFiberToSchedule.fiber != nullptr)
            {
                gSleepingFiberQueue.push(currentFiber->sleepingFiberToSchedule);
                currentFiber->sleepingFiberToSchedule.fiber = nullptr;
            }

            const bool haveAnySleepingFibers = gSleepingFiberQueue.try_pop(sleepingFiber);

            if (haveAnySleepingFibers)
            {
                if (LoadCounter(sleepingFiber.counterHandle) == sleepingFiber.condition)
                {
                    gFreeFiberQueue.push(currentFiber->index);
                    SwitchToAnotherFiber(sleepingFiber.fiber->handle);
                    continue;
                }
                else
                {
                    gSleepingFiberQueue.push(sleepingFiber);
                }
            }

            // Last in first out (LIFO)
            if (gJobQueue.try_pop(job))
            {
                if (job.decl.jobFunc)
                {
                    job.decl.jobFunc(job.decl.data);
                }
                FetchSubCounter(job.counterHandle);
            }
            else if (!haveAnySleepingFibers)
            {
                const uint32 key = GetCurrentThreadID();
                SemaphoreWait(gSemaphores[gThreadIdSemaphoreLUT[key]].handle);
            }
        }

        uint32 threadID = GetCurrentThreadID();

        uint32 threadIndex = 0;
        for (uint32 i = 0; i < gWorkerThreadCount; i++)
        {
            if (gWorkerThreadIDs[i] == threadID)
            {
                threadIndex = i;
                break;
            }
        }

        uint64 fiberHandle = ((Fiber*)GetCurrentFiberData()->userData)->handle;
        if (fiberHandle != gFibers[threadIndex].handle)
        {
            SwitchToAnotherFiber(gFibers[threadIndex].handle);
        }

        ConvertCurrentFiberToThread();
    }

    static void WorkerThreadEntry(void* userData)
    {
        WorkerThreadUserData* workerThreadUserData = (WorkerThreadUserData*)userData;

        uint32 workerThreadIndex = workerThreadUserData->workerThreadIndex;

        gFiberData[workerThreadIndex].fiberEntry = FiberEntry;
        gFiberData[workerThreadIndex].userData = &gFibers[workerThreadIndex];

        Fiber& fiber = gFibers[workerThreadIndex];
        fiber.index = workerThreadIndex;
        fiber.handle = ConvertCurrentThreadToFiber(&gFiberData[workerThreadIndex]);
        fiber.sleepingFiberToSchedule.fiber = nullptr;

        workerThreadUserData->bootAtomic->fetch_sub(1);

        OPTICK_THREAD("WorkerThread");

        FiberEntry(&gFibers[workerThreadIndex]);
    }

    void JobSystemInit(uint32 numWorkerThreads, uint32 numFibers, uint32 fiberStackSize)
    {
        ASSERT(!gInitialized);
        ASSERT(numWorkerThreads <= JOB_SYSTEM_MAX_NUM_WORKER_THREADS);
        ASSERT(numFibers >= numWorkerThreads && (numFibers & (numFibers - 1)) == 0 && numFibers <= JOB_SYSTEM_MAX_NUM_FIBERS);

        std::atomic<uint32> bootAtomic;
        bootAtomic.store(numWorkerThreads);

        for (uint32 workerThreadIndex = 0; workerThreadIndex < numWorkerThreads; workerThreadIndex++)
        {
            gWorkerThreadUserData[workerThreadIndex].workerThreadIndex = workerThreadIndex;
            gWorkerThreadUserData[workerThreadIndex].bootAtomic = &bootAtomic;

            gThreadData[workerThreadIndex].threadEntry = WorkerThreadEntry;
            gThreadData[workerThreadIndex].userData = &gWorkerThreadUserData[workerThreadIndex];

            wchar description[100];
            wsprintf(description, L"JobSystem::WorkerThread %d", workerThreadIndex);
            gWorkerThreads[workerThreadIndex] = CreateWokerThread(0, &gThreadData[workerThreadIndex], description);

            gSemaphores[workerThreadIndex].handle = CreateSemaphoreEXT(0);

            const uint32 key = gWorkerThreads[workerThreadIndex].id;
            gThreadIdSemaphoreLUT.emplace(key, workerThreadIndex);
        }
        gWorkerThreadCount = numWorkerThreads;

        while (bootAtomic.load(std::memory_order_acquire) != 0)
        {
            SuspendCurrentThread(0.01f);
        }

        for (uint32 i = 0; i < JOB_SYSTEM_MAX_NUM_JOBS; i++)
        {
            gAtomicCounters[i].index = i;
            gFreeCounterQueue.push(i);
        }

        Fiber fiber = {};
        for (uint32 fiberIndex = gWorkerThreadCount; fiberIndex < gFiberCount; fiberIndex++)
        {
            gFiberData[fiberIndex].fiberEntry = FiberEntry;
            gFiberData[fiberIndex].userData = &gFibers[fiberIndex];

            fiber.handle = CreateFiber(fiberStackSize, &gFiberData[fiberIndex]);
            fiber.index = fiberIndex;

            gFibers[fiberIndex] = fiber;
            gFreeFiberQueue.push(fiberIndex);
        }
        gFiberCount = numFibers;
        gNextWorkerThreadIndex = 0;

        gInitialized.store(true, std::memory_order_release);

        HE_LOG_INFO("Job system init.");
    }

    void JobSystemExit()
    {
        ASSERT(gInitialized);
        HE_LOG_INFO("Job system exit.");
        // TODO
    }

    JobSystemAtomicCounterHandle JobSystemRunJobs(JobSystemJobDecl* jobDecls, uint32 numJobs)
    {
        uint32 freeCounterIndex;
        while (!FindFreeCounter(freeCounterIndex));

        JobSystemAtomicCounterHandle freeCounter = freeCounterIndex + 1;
        StoreCounter(freeCounter, numJobs);

        Job job = {};
        job.counterHandle = freeCounter;

        for (uint32 jobIndex = 0; jobIndex < numJobs; jobIndex++)
        {
            job.decl = jobDecls[jobIndex];

            gJobQueue.push(job);

            uint32 workerThreadIndex = gNextWorkerThreadIndex.fetch_add(1);
            SemaphoreAdd(gSemaphores[workerThreadIndex % gWorkerThreadCount].handle, 1);
        }

        return freeCounter;
    }

    void JobSystemWaitForCounter(JobSystemAtomicCounterHandle counterHandle, uint32 condition)
    {
        if (LoadCounter(counterHandle) != condition)
        {
            uint32 freeFiberIndex;
            while (!FindFreeFiber(freeFiberIndex));

            Fiber* nextFiber = &gFibers[freeFiberIndex];
            Fiber* currentFiber = (Fiber*)(GetCurrentFiberData()->userData);

            nextFiber->sleepingFiberToSchedule = {
                condition,
                counterHandle,
                currentFiber
            };

            SwitchToAnotherFiber(nextFiber->handle);
        }
    }

    void JobSystemWaitForCounterAndFree(JobSystemAtomicCounterHandle counterHandle, uint32 condition)
    {
        JobSystemWaitForCounter(counterHandle, condition);
        FreeCounter(counterHandle);
    }

    void JobSystemWaitForCounterAndFreeWithoutFiber(JobSystemAtomicCounterHandle counterHandle)
    {
        while (LoadCounter(counterHandle) != 0)
        {
            SuspendCurrentThread(0.01f);
        }
        FreeCounter(counterHandle);
    }
}
