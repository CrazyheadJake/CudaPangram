cmake_minimum_required(VERSION 3.26.3)


find_package(CUDAToolkit REQUIRED)  # Already finds headers/libs

# Allow switching between Debug and Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})

# Clear cached flags
# set(CMAKE_CUDA_FLAGS "" CACHE STRING "Release compiler flags" FORCE)
# set(CMAKE_CUDA_FLAGS_RELEASE "" CACHE STRING "Release compiler flags" FORCE)
# set(CMAKE_CUDA_FLAGS_DEBUG "" CACHE STRING "Release compiler flags" FORCE)
# set(CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "Release compiler flags" FORCE)
# set(CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "Release compiler flags" FORCE)

project(CudaPangram LANGUAGES CXX CUDA)

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

# Set cmd line options
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --diag-suppress=177   # Suppress warnings
        --ptxas-options=-v
        --maxrregcount=0
        -O0
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        /Od
    >
)

message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "BUILD TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CUDA Release flags: ${CMAKE_CUDA_FLAGS_RELEASE}")