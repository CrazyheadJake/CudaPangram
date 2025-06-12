#include "kernel.cuh"
#include <cuda_runtime.h>   // CUDA runtime API
#include <device_launch_parameters.h> // Optional: threadIdx, blockIdx, etc.
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <string>
#include <vector>
#include "filteredWords.inl"

namespace fs = std::filesystem;

const int THREADSPERBLOCK = 64;
const int WORDS = 14855;
const int WORDS_SIZE =  WORDS * sizeof(uint32_t);
const int FILTEREDWORDS = 8401;
const int FILTEREDWORDS_SIZE = FILTEREDWORDS * sizeof(uint32_t);
const int BLOCKS = ((FILTEREDWORDS * FILTEREDWORDS) / THREADSPERBLOCK) + 1;
const int SOLUTIONS_SIZE = 80000000;

__device__ int solutionIdx = 0;

const int PANGRAMMASK = 67108863;       //stringToMask("abcdefghijklmnopqrstuvwxyz")

struct Solution {
    uint32_t words[6];
};


// CUDA kernel
__global__ void Kernel(uint32_t* dMasks, Solution* dSolutions) {
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int start0 = gid / FILTEREDWORDS;
    unsigned int start1 = start0 + gid % FILTEREDWORDS;
    if (start0 >= FILTEREDWORDS || start1 >= FILTEREDWORDS)
        return;
    Solution sln = {start0, start1, 0, 0, 0, 0};
    uint32_t masks[6] = {dMasks[sln.words[0]], dMasks[sln.words[0]] | dMasks[sln.words[1]], 0, 0, 0, 0};
    // printf("Starting thread idx: %d\n", gid);
    int letterCount;
    
    // for (sln.words[1] = sln.words[0] + 1; sln.words[1] < FILTEREDWORDS; sln.words[1]++) {
    //     masks[1] = dMasks[sln.words[1]] | masks[0];
    //     // Don't need to check letter count, all possible words here lead to valid solutions
    //     // printf("Current word at depth 1: %d\n", sln.words[1]);

    //     // Exit early to check if output works
    //     if (sln.words[1] == 5)
    //         return;

        for (sln.words[2] = sln.words[1] + 1; sln.words[2] < FILTEREDWORDS; sln.words[2]++) {
            masks[2] = dMasks[sln.words[2]] | masks[1];
            letterCount = __popc(masks[2]);     // Count how many bits in the mask are set
            // printf("Letter Count: %d\n", letterCount);
            if (letterCount < 11)
                continue;

            for (sln.words[3] = sln.words[2] + 1; sln.words[3] < FILTEREDWORDS; sln.words[3]++) {
                masks[3] = dMasks[sln.words[3]] | masks[2];
                letterCount = __popc(masks[3]);     // Count how many bits in the mask are set
                    if (letterCount < 16)
                        continue;

                for (sln.words[4] = sln.words[3] + 1; sln.words[4] < FILTEREDWORDS; sln.words[4]++) {
                    masks[4] = dMasks[sln.words[4]] | masks[3];
                    letterCount = __popc(masks[4]);     // Count how many bits in the mask are set
                    if (letterCount < 21)
                        continue;

                    for (sln.words[5] = sln.words[4] + 1; sln.words[5] < FILTEREDWORDS; sln.words[5]++) {
                        masks[5] = dMasks[sln.words[5]] | masks[4];

                        // Print out the state of the program
                        // printf("Mask: %d, \t\tWords: %d, %d, %d, %d, %d, %d\n", masks[5], sln.words[0], sln.words[1], sln.words[2], sln.words[3], sln.words[4], sln.words[5]);

                        if (masks[5] == PANGRAMMASK) {
                            // Atomically increment the counter to reserve the index
                            int idx = atomicAdd(&solutionIdx, 1);

                            dSolutions[idx] = sln;
                            printf("%d, %d, %d, %d, %d, %d\n", sln.words[0], sln.words[1], sln.words[2], sln.words[3], sln.words[4], sln.words[5]);

                        }
                    }
                }
            }
        }
    // }
    // printf("Thread idx %d done!\n", gid);
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

int main() {
    printDeviceInfo();
    
    uint32_t hMasks[FILTEREDWORDS];
    Solution* hSolutions = new Solution[SOLUTIONS_SIZE]; // Heap allocated
    getMasks(hMasks, noAnagramsWords);

    uint32_t *dMasks;
    Solution* dSolutions;
    cudaMalloc((void **)(&dMasks), FILTEREDWORDS_SIZE);
    cudaMalloc((void **)(&dSolutions), SOLUTIONS_SIZE * sizeof(Solution));

    cudaMemcpy(dMasks, hMasks, FILTEREDWORDS_SIZE, cudaMemcpyHostToDevice);

    std::cout << "Calling kernel" << std::endl;

    Kernel<<<BLOCKS, THREADSPERBLOCK>>>(dMasks, dSolutions);
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