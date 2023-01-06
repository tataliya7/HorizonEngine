//#define malloc(size) printf("Allocated %d bytes.\n", size); malloc(size);
//#define free(ptr) printf("Free %d bytes.\n", sizeof(ptr)); free(ptr);
//inline void* operator new(size_t size)
//{
//	printf("Allocated %zd bytes.\n", size);
//	return malloc(size);
//}
//
//inline void operator delete(void* ptr, size_t size)
//{
//	printf("Free %zd bytes.\n", size);
//	free(ptr);
//}

//namespace HE
//{
//	extern uint64 gMainThreadID;
//	extern bool IsInMainThread();
//	namespace CoreGlobals
//	{
//		extern void Init();
//		extern void Shutdown();
//	}
//}