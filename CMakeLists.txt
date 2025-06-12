cmake_minimum_required(VERSION 3.26.3)

project(CudaPangram LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)  # Already finds headers/libs

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add your sources
add_executable(${PROJECT_NAME}
    src/main.cu
)

# Tell CMake where to find CUDA headers for both .cu and .cpp files
target_include_directories(${PROJECT_NAME}
    PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    ${CUDAToolkit_INCLUDE_DIRS}      # <<< this line is key
)

# Link the CUDA libraries (e.g., cudart) — needed for host calls like cudaMalloc
target_link_libraries(${PROJECT_NAME}
    PUBLIC
    CUDA::cudart                    # safer than hardcoded paths
)

# Suppress warnings
target_compile_options(${PROJECT_NAME}
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=177>     # Ignore "variable not used" warnings
)

message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
