#ifndef KERNEL_CUH
#define KERNEL_CUH
#include <cuda_runtime.h>   // CUDA runtime API
#include <device_launch_parameters.h> // Optional: threadIdx, blockIdx, etc.

// Declare your kernel
__global__ void Kernel();
__host__ void RunKernel();

#endif
