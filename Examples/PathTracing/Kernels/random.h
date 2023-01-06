#pragma once

// TODO: remove this
static __host__ __device__ __inline__ unsigned int tea(unsigned int val0, unsigned int val1)
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;
    for (unsigned int n = 0; n < 4; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
        v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }
    return v0;
}

static __host__ __device__ __inline__ unsigned int lcg(unsigned int& seed)
{   
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    seed = (LCG_A * seed + LCG_C);
    return seed & 0x00FFFFFF;
}

static __host__ __device__ __inline__ float randf(unsigned int& seed)
{
    return ((float)lcg(seed) / (float)0x01000000);
}
