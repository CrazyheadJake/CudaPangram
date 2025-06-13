#include "kernel.cuh"
#include <cuda_runtime.h>   // CUDA runtime API
#include <device_launch_parameters.h> // Optional: threadIdx, blockIdx, etc.
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <string>
#include <vector>
#include <cassert>
#include "filteredWords.inl"

namespace fs = std::filesystem;

const int THREADSPERBLOCK = 64;
const int FILTEREDWORDS = 8401;
const int FILTEREDWORDS_SIZE = FILTEREDWORDS * sizeof(uint32_t);
const int BLOCKS = 16384;
const int STRIDE = BLOCKS*THREADSPERBLOCK;
const int SOLUTIONS_SIZE = 80000000;

__device__ uint32_t solutionIdx = 0;

const int PANGRAMMASK = 67108863;       //stringToMask("abcdefghijklmnopqrstuvwxyz")

struct Solution {
    uint32_t words[6];
};


// CUDA kernel
__global__ void Kernel(uint32_t* dMasks, Solution* dSolutions) {
    uint32_t gridId = (blockIdx.x*blockDim.x + threadIdx.x);
    uint32_t idx;
    uint32_t w2, w3, w4, w5;
    uint32_t m1, m2, m3, m4, m5;
    uint32_t start0, start1;
    uint32_t gid;
    for (gid = gridId; gid < FILTEREDWORDS*FILTEREDWORDS; gid += STRIDE) {
    start0 = gid / FILTEREDWORDS;
    start1 = start0 + gid % FILTEREDWORDS;
   
    if (start0 >= FILTEREDWORDS || start1 >= FILTEREDWORDS)
        continue;

    
    m1 = dMasks[start0] | dMasks[start1];    
    for (w2 = start1 + 1; w2 < FILTEREDWORDS; w2++) {
        m2 = dMasks[w2] | m1;
        if (__popc(m2) < 11)    // Count how many bits in the mask are set 
            continue;

        for (w3 = w2 + 1; w3 < FILTEREDWORDS; w3++) {
            m3 = dMasks[w3] | m2;
            if (__popc(m3) < 16)    // Count how many bits in the mask are set 
                continue;

            for (w4 = w3 + 1; w4 < FILTEREDWORDS; w4++) {
                m4 = dMasks[w4] | m3;
                if (__popc(m4) < 21)    // Count how many bits in the mask are set 
                    continue;

                for (w5 = w4 + 1; w5 < FILTEREDWORDS; w5++) {
                    if ((dMasks[w5] | m4) == PANGRAMMASK) {
                        // Atomically increment the counter to reserve the index
                        idx = atomicAdd(&solutionIdx, 1);
                        dSolutions[idx] = {start0, start1, w2, w3, w4, w5};
                        // printf("%d, %d, %d, %d, %d, %d\t\tMasks: %d, %d, %d, %d, %d\n", w0, w1, w2, w3, w4, w5, dMasks[w1], dMasks[w2], dMasks[w3], dMasks[w4], dMasks[w5]);
                        printf("Solutions: %d\n", idx + 1);
                    }
                }
            }
        }
    }
}
}

uint32_t stringToMask(const std::string& str) {
    uint32_t mask = 0;
    for (char c : str) {
        mask |= (1 << (c - 'a'));
    }
    return mask;
}

void readData(uint32_t* hWords) {
    std::ifstream input;
    input.open("filteredWords.txt");

    if (!input.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        exit(1); // Indicate an error
    }

    std::string line;
    int i = 0;
    while (std::getline(input, line)) {
        // Process the 'line' string
        hWords[i] = stringToMask(line);
        i++;
    }
    input.close();
}

void writeSolutions(Solution* hSolutions, int numSolutions, const std::string& filename) {
    std::ofstream output;
    output.open(filename);
    if (!output.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1); // Indicate an error
    }

    for (int i = 0; i < numSolutions; i++) {
        output << hSolutions[i].words[0] << ", ";
        output << hSolutions[i].words[1] << ", ";
        output << hSolutions[i].words[2] << ", ";
        output << hSolutions[i].words[3] << ", ";
        output << hSolutions[i].words[4] << ", ";
        output << hSolutions[i].words[5] << ", " << std::endl;
    }
    output.close();
}

void writeMasks(uint32_t* hMasks, const std::string& filename) {
    std::ofstream output;
    output.open(filename);
    if (!output.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1); // Indicate an error
    }

    for (int i = 0; i < FILTEREDWORDS; i++) {
        output << hMasks[i] << std::endl;

    }
    output.close();

}

void getMasks(uint32_t* hMasks, std::string* words) {
    for (int i = 0; i < FILTEREDWORDS; i++) {
        hMasks[i] = stringToMask(words[i]);
    }
}


void printDeviceInfo() {
    int device = 0;
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    std::cout << "GPU name: " << props.name << "\n";
    std::cout << "Streaming Multiprocessors (SMs): " << props.multiProcessorCount << "\n";
    std::cout << "Compute Capability: " << props.major << "." << props.minor << "\n";
    std::cout << "Number of CUDA Cores: " << props.multiProcessorCount * 128 << "\n";
}

void checkCudaErrors() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

int main() {
    printDeviceInfo();
    
    uint32_t hMasks[FILTEREDWORDS];
    Solution* hSolutions = new Solution[SOLUTIONS_SIZE]; // Heap allocated
    getMasks(hMasks, noAnagramsWords);
    assert(hMasks[0] = 153 && hMasks[1] == 2305);

    uint32_t* dMasks;
    Solution* dSolutions;
    cudaMalloc((void **)(&dSolutions), SOLUTIONS_SIZE * sizeof(Solution));
    cudaMalloc((void **)(&dMasks), FILTEREDWORDS_SIZE);
    // cudaMemcpyToSymbol(dMasks, hMasks, FILTEREDWORDS_SIZE, 0, cudaMemcpyHostToDevice);
    cudaMemcpy(dMasks, hMasks, FILTEREDWORDS_SIZE, cudaMemcpyHostToDevice);
    checkCudaErrors();

    std::cout << "Calling kernel" << std::endl;
    Kernel<<<BLOCKS, THREADSPERBLOCK>>>(dMasks, dSolutions);
    checkCudaErrors();
    cudaDeviceSynchronize();

    std::cout << "Kernel complete" << std::endl;

    // Copy result back to host
    int numSolutions;
    cudaMemcpyFromSymbol(&numSolutions, solutionIdx, sizeof(int));
    cudaMemcpy(hSolutions, dSolutions, SOLUTIONS_SIZE * sizeof(Solution), cudaMemcpyDeviceToHost);

    writeSolutions(hSolutions, numSolutions, "solutions.txt");

    delete hSolutions;

    return 0;
}