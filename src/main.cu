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

    // Grid stride looping to use fewer threads while still covering search space
    for (gid = gridId; gid < ALLWORDS*ALLWORDS; gid += STRIDE) {
        w0 = gid / ALLWORDS;
        w1 = w0 + gid % ALLWORDS + 1;

        // If out of bounds, skip
        if (w0 >= ALLWORDS || w1 >= ALLWORDS)
            continue;

        m1 = dMasks[w0] | dMasks[w1];
        // If cache value = x, then you must check all words <= x 
        // Check the cache, skip if we know there are know solutions
        cacheValue = dCache[m1 | LVL_1_MASK];
        if (w1 > cacheValue && cacheValue != CACHE_HAS_SOLUTION) {
            continue;
        }

        for (w2 = w1 + 1; w2 < ALLWORDS; w2++) {
            m2 = dMasks[w2] | m1;
            // Count how many bits in the mask are set 
            if (__popc(m2) < 11)
                continue;
            

            // Check the cache, skip if we know there are know solutions
            cacheValue = dCache[m2 | LVL_2_MASK];
            if (w2 > cacheValue && cacheValue != CACHE_HAS_SOLUTION) {
                continue;
            }

            for (w3 = w2 + 1; w3 < ALLWORDS; w3++) {
                m3 = dMasks[w3] | m2;
                // Count how many bits in the mask are set 
                if (__popc(m3) < 16)
                    continue;

                // Check the cache, skip if we know there are know solutions
                cacheValue = dCache[m3 | LVL_3_MASK];
                if (w3 > cacheValue && cacheValue != CACHE_HAS_SOLUTION) {
                    continue;
                }

                for (w4 = w3 + 1; w4 < ALLWORDS; w4++) {
                    m4 = dMasks[w4] | m3;
                    // Count how many bits in the mask are set 
                    if (__popc(m4) < 21)
                        continue;
                    
                    // Check the cache, skip if we know there are know solutions
                    cacheValue = dCache[m4 | LVL_4_MASK];
                    if (w4 > cacheValue && cacheValue != CACHE_HAS_SOLUTION) {
                        continue;
                    }

                    for (w5 = w4 + 1; w5 < ALLWORDS; w5++) {
                        if ((dMasks[w5] | m4) == PANGRAMMASK) {
                            // Tell all layers of relevant cache that a solution has been found
                            dCache[m1 | LVL_1_MASK] = CACHE_HAS_SOLUTION;
                            dCache[m2 | LVL_2_MASK] = CACHE_HAS_SOLUTION;
                            dCache[m3 | LVL_3_MASK] = CACHE_HAS_SOLUTION;
                            dCache[m4 | LVL_4_MASK] = CACHE_HAS_SOLUTION;

                            // Atomically increment the counter to reserve the index
                            idx = atomicAdd(&solutionIdx, 1);
                            dSolutions[idx] = {w0, w1, w2, w3, w4, w5};
                            if (idx % 100000 == 99999)
                                printf("Solutions: %d\n", idx + 1);
                        }
                    }
                    // For each depth, if there was no solution found in the search tree, mark the branch as skippable
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

// Convert string to a bitmask of letters used in the string
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
        exit(1);
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

// Write solutions in the form of integers to a file, NOTE THAT THIS IS EXCRUCIATINGLY SLOW AND UNOPTIMIZED
void writeSolutions(Solution* hSolutions, int numSolutions, const std::string& filename) {
    std::ofstream output;
    output.open(filename);
    if (!output.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
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
        exit(1);
    }
    std::cout << "Sorting pangrams" << std::endl;
    std::ostringstream buffer;
    std::stringstream temp;
    std::string word;
    std::vector<std::string> words;
    for (int i = 0; i < numSolutions; i++) {
        for (int j = 0; j < 6; j++) {
            word = allWords[hSolutions[i].words[j]];
            std::transform(word.begin(), word.end(), word.begin(),
                   [](unsigned char c) { return std::toupper(c); });
            temp << word;
            temp << (j < 5 ? " " : "\n");
        }
        words.push_back(temp.str());
        temp.str("");
    }
    // Sort the pangrams alphabetically
    std::sort(words.begin(), words.end());
    for (int i = 0; i < numSolutions; i++) {
        buffer << words[i];
    }
    std::cout << "Writing to file" << std::endl;
    output << buffer.str();  // One big write
    output.close();
}

void writeMasks(uint32_t* hMasks, const std::string& filename) {
    std::ofstream output;
    output.open(filename);
    if (!output.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    for (int i = 0; i < ALLWORDS; i++) {
        output << hMasks[i] << std::endl;

    }
    output.close();

}

void getMasks(uint32_t* hMasks, std::string* words) {
    for (int i = 0; i < ALLWORDS; i++) {
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
    
    // Initialize variables on the host (CPU)
    uint32_t hMasks[ALLWORDS];
    Solution* hSolutions = new Solution[SOLUTIONS_SIZE];    // Too big for the stack
    getMasks(hMasks, allWords);
    assert(hMasks[0] == 153 && hMasks[1] == 2305);

    // Initialize and allocate memory on the device (GPU)
    uint32_t* dMasks;
    Solution* dSolutions;
    uint16_t* dCache;
    cudaMalloc((void **)(&dSolutions), SOLUTIONS_SIZE * sizeof(Solution));
    cudaMalloc((void **)(&dMasks), ALLWORDS_SIZE);
    cudaMalloc((void **)(&dCache), CACHE_SIZE * sizeof(uint16_t));
    checkCudaErrors();

    // Copy values over to the device from the host
    cudaMemcpy(dMasks, hMasks, ALLWORDS_SIZE, cudaMemcpyHostToDevice);
    cudaMemset(dCache, CACHE_EMPTY, CACHE_SIZE * sizeof(uint16_t));
    checkCudaErrors();

    // Launch the kernel
    std::cout << "Calling kernel" << std::endl;
    Kernel<<<BLOCKS, THREADSPERBLOCK>>>(dMasks, dSolutions, dCache);
    checkCudaErrors();
    cudaDeviceSynchronize();
    std::cout << "Kernel complete" << std::endl;

    // Copy data from the device back to the host
    int numSolutions;
    cudaMemcpyFromSymbol(&numSolutions, solutionIdx, sizeof(int));
    cudaMemcpy(hSolutions, dSolutions, SOLUTIONS_SIZE * sizeof(Solution), cudaMemcpyDeviceToHost);

    // Write output to file
    writeWords(hSolutions, numSolutions, "solutions/final_solutions.txt");

    delete hSolutions;
    return 0;
}