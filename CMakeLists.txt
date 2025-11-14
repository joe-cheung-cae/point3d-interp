cmake_minimum_required(VERSION 3.18)
project(point3d_interp LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

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

# Precision settings
if(USE_DOUBLE_PRECISION)
    add_definitions(-DUSE_DOUBLE_PRECISION)
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files (only include existing files)
set(SOURCES
    src/error_codes.cpp
    src/data_loader.cpp
    src/grid_structure.cpp
    src/cpu_interpolator.cpp
)

set(CUDA_SOURCES
    src/cuda_interpolator.cu
    src/memory_manager.cu
    src/api.cu
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
