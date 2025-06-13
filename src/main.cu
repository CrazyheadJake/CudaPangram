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
#include <sstream>

#include "filteredWords.inl"
#include "allWords.inl"

namespace fs = std::filesystem;

const int THREADSPERBLOCK = 64;
const int ALLWORDS = 14855;
const int ALLWORDS_SIZE = ALLWORDS * sizeof(uint32_t);
// const int FILTEREDWORDS = 8401;
// const int FILTEREDWORDS_SIZE = FILTEREDWORDS * sizeof(uint32_t);
const int FILTEREDWORDS = ALLWORDS;
const int FILTEREDWORDS_SIZE = ALLWORDS_SIZE;
const int BLOCKS = 16384;
const int STRIDE = BLOCKS*THREADSPERBLOCK;
const int SOLUTIONS_SIZE = 80000000;

const uint32_t LVL_1_MASK = 1 << 26;
const uint32_t LVL_2_MASK = 1 << 27;
const uint32_t LVL_3_MASK = 1 << 28;
const uint32_t LVL_4_MASK = 1 << 29;

const uint32_t CACHE_SIZE = 1 << 30;
const uint16_t CACHE_EMPTY = UINT16_MAX;
const uint16_t CACHE_NOSOLUTIONS = 1;
const uint16_t CACHE_HAS_SOLUTION = UINT16_MAX-1;


__device__ uint32_t solutionIdx = 0;
__device__ uint32_t cacheHits1 = 0;
__device__ uint32_t cacheHits2 = 0;
__device__ uint32_t cacheHits3 = 0;
__device__ uint32_t cacheHits4 = 0;

const int PANGRAMMASK = (1 << 26) - 1;       //stringToMask("abcdefghijklmnopqrstuvwxyz")

struct Solution {
    uint32_t words[6];
};


// CUDA kernel
__global__ void Kernel(uint32_t* dMasks, Solution* dSolutions, uint16_t* dCache) {
    uint32_t gridId = (blockIdx.x*blockDim.x + threadIdx.x);
    uint32_t idx;
    uint32_t w0, w1, w2, w3, w4, w5;
    uint32_t m1, m2, m3, m4;
    uint32_t gid;
    uint16_t cacheValue;
    for (gid = gridId; gid < FILTEREDWORDS*FILTEREDWORDS; gid += STRIDE) {
    w0 = gid / FILTEREDWORDS;
    w1 = w0 + gid % FILTEREDWORDS;
   
    if (w0 >= FILTEREDWORDS || w1 >= FILTEREDWORDS)
        continue;

    m1 = dMasks[w0] | dMasks[w1];

    // If cache value = x, then you must check all words <= x 
    // Check the cache, skip if we know there are know solutions
    cacheValue = dCache[m1 | LVL_1_MASK];
    if (w1 > cacheValue && cacheValue != CACHE_HAS_SOLUTION) {
        continue;
    }


    for (w2 = w1 + 1; w2 < FILTEREDWORDS; w2++) {
        m2 = dMasks[w2] | m1;
        // Count how many bits in the mask are set 
        if (__popc(m2) < 11)
            continue;


        // Check the cache, skip if we know there are know solutions
        cacheValue = dCache[m2 | LVL_2_MASK];
        if (w2 > cacheValue && cacheValue != CACHE_HAS_SOLUTION) {
            continue;
        }

        for (w3 = w2 + 1; w3 < FILTEREDWORDS; w3++) {
            m3 = dMasks[w3] | m2;
            // Count how many bits in the mask are set 
            if (__popc(m3) < 16)
                continue;

            // Check the cache, skip if we know there are know solutions
            cacheValue = dCache[m3 | LVL_3_MASK];
            if (w3 > cacheValue && cacheValue != CACHE_HAS_SOLUTION) {
                continue;
            }

            for (w4 = w3 + 1; w4 < FILTEREDWORDS; w4++) {
                m4 = dMasks[w4] | m3;
                // Count how many bits in the mask are set 
                if (__popc(m4) < 21)
                    continue;
                
                // Check the cache, skip if we know there are know solutions
                cacheValue = dCache[m4 | LVL_4_MASK];
                if (w4 > cacheValue && cacheValue != CACHE_HAS_SOLUTION) {
                    continue;
                }

                for (w5 = w4 + 1; w5 < FILTEREDWORDS; w5++) {
                    if ((dMasks[w5] | m4) == PANGRAMMASK) {
                        // Tell all layers of relevant cache that a solution has been found
                        dCache[m1 | LVL_1_MASK] = CACHE_HAS_SOLUTION;
                        dCache[m2 | LVL_2_MASK] = CACHE_HAS_SOLUTION;
                        dCache[m3 | LVL_3_MASK] = CACHE_HAS_SOLUTION;
                        dCache[m4 | LVL_4_MASK] = CACHE_HAS_SOLUTION;

                        // Atomically increment the counter to reserve the index
                        idx = atomicAdd(&solutionIdx, 1);
                        dSolutions[idx] = {w0, w1, w2, w3, w4, w5};
                        // printf("%d, %d, %d, %d, %d, %d\t\tMasks: %d, %d, %d, %d, %d\n", w0, w1, w2, w3, w4, w5, dMasks[w1], dMasks[w2], dMasks[w3], dMasks[w4], dMasks[w5]);
                        printf("Solutions: %d, Cache Hits: %d, %d, %d, %d\n", idx + 1, cacheHits1, cacheHits2, cacheHits3, cacheHits4);
                    }
                }
                if (dCache[m4 | LVL_4_MASK] > w4 && dCache[m4 | LVL_4_MASK] != CACHE_HAS_SOLUTION) {
                    dCache[m4 | LVL_4_MASK] = w4;
                }
            }
            if (dCache[m3 | LVL_3_MASK] > w3 && dCache[m3 | LVL_3_MASK] != CACHE_HAS_SOLUTION) {
                dCache[m3 | LVL_3_MASK] = w3;
            }
        }
        if (dCache[m2 | LVL_2_MASK] > w2 && dCache[m2 | LVL_2_MASK] != CACHE_HAS_SOLUTION) {
            dCache[m2 | LVL_2_MASK] = w2;
        }
    }
    if (dCache[m1 | LVL_1_MASK] > w1 && dCache[m1 | LVL_1_MASK] != CACHE_HAS_SOLUTION) {
        dCache[m1 | LVL_1_MASK] = w1;
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

void writeWords(Solution* hSolutions, int numSolutions, const std::string& filename) {
    std::ofstream output;
    output.open(filename);
    if (!output.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1); // Indicate an error
    }

    std::ostringstream buffer;
    for (int i = 0; i < numSolutions; ++i) {
        for (int j = 0; j < 6; ++j) {
            buffer << allWords[hSolutions[i].words[j]];
            buffer << (j < 5 ? "," : "\n");
        }
    }
    output << buffer.str();  // One big write
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
    getMasks(hMasks, allWords);
    assert(hMasks[0] = 153 && hMasks[1] == 2305);

    uint32_t* dMasks;
    Solution* dSolutions;
    uint16_t* dCache;
    cudaMalloc((void **)(&dSolutions), SOLUTIONS_SIZE * sizeof(Solution));
    cudaMalloc((void **)(&dMasks), FILTEREDWORDS_SIZE);
    cudaMalloc((void **)(&dCache), CACHE_SIZE * sizeof(uint16_t));
    checkCudaErrors();

    // cudaMemcpyToSymbol(dMasks, hMasks, FILTEREDWORDS_SIZE, 0, cudaMemcpyHostToDevice);
    cudaMemcpy(dMasks, hMasks, FILTEREDWORDS_SIZE, cudaMemcpyHostToDevice);
    cudaMemset(dCache, CACHE_EMPTY, CACHE_SIZE * sizeof(uint16_t));
    checkCudaErrors();

    std::cout << "Calling kernel" << std::endl;
    Kernel<<<BLOCKS, THREADSPERBLOCK>>>(dMasks, dSolutions, dCache);
    checkCudaErrors();
    cudaDeviceSynchronize();

    std::cout << "Kernel complete" << std::endl;

    // Copy result back to host
    int numSolutions;
    cudaMemcpyFromSymbol(&numSolutions, solutionIdx, sizeof(int));
    cudaMemcpy(hSolutions, dSolutions, SOLUTIONS_SIZE * sizeof(Solution), cudaMemcpyDeviceToHost);
    // uint8_t* hCache = new uint8_t[CACHE_SIZE];
    // cudaMemcpy(hCache, dCache, CACHE_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    // uint32_t cachedSolutions = 0;
    // std::cout << "Counting solutions";
    // for (uint32_t i = 0; i < CACHE_SIZE; i++) {
    //     if (hCache[i] == CACHE_HAS_SOLUTION) {
    //         cachedSolutions++;
    //     }
    // }

    writeWords(hSolutions, numSolutions, "allsolutions.txt");

    delete hSolutions;

    return 0;
}