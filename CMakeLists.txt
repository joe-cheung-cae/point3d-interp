cmake_minimum_required(VERSION 3.25.0 FATAL_ERROR)

# Set the CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Choose the CUDA architectures")
set_property(CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY STRINGS native all all-major)

project(point3d_interp LANGUAGES CXX CUDA)

# Set CMAKE build type
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile command
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Enable clang-tidy for C++ targets if available
find_program(CLANG_TIDY_EXE clang-tidy)

if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-p;${CMAKE_BINARY_DIR}")
endif()

# Options
option(USE_DOUBLE_PRECISION "Use double precision" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(ENABLE_SIMD "Enable SIMD optimizations" ON)

# Precision settings
if(USE_DOUBLE_PRECISION)
    add_definitions(-DUSE_DOUBLE_PRECISION)
endif()

# SIMD optimizations
if(ENABLE_SIMD)
    add_definitions(-DENABLE_SIMD)

    if(MSVC)
        # MSVC SIMD flags
        add_compile_options(/arch:AVX2)
    else()
        # GCC/Clang SIMD flags
        add_compile_options(-mavx2 -mfma)
    endif()
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Copy data files to build directory for examples
file(COPY ${PROJECT_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# Source files (only include existing files)
set(SOURCES
    src/error_codes.cpp
    src/data_loader.cpp
    src/grid_structure.cpp
    src/cpu_interpolator.cpp
    src/spatial_grid.cpp
    src/unstructured_interpolator.cpp
    src/kd_tree.cpp
    src/exporter.cpp
)

set(CUDA_SOURCES
    src/api.cu
    src/cuda_interpolator.cu
    src/memory_manager.cu
    src/interpolator_adapters.cu
    src/interpolator_factory.cu
)

# Static library
add_library(point3d_interp STATIC ${SOURCES} ${CUDA_SOURCES})

# CUDA compile options
target_compile_options(point3d_interp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    --expt-relaxed-constexpr
    >
)

# Link CUDA libraries
target_link_libraries(point3d_interp PUBLIC cudart)

# Install rules
install(TARGETS point3d_interp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/point3d_interp
    DESTINATION include
)

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
