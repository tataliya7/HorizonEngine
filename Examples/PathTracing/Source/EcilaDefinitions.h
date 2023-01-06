#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define FORCEINLINE __forceinline__
#    define CUDA_HOST_DEVICE __device__
#else
#    define FORCEINLINE inline
#    define CUDA_HOST_DEVICE __host__
#endif
