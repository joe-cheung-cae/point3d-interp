cmake_minimum_required(VERSION 3.18)
project(point3d_interp LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# 选项
option(USE_DOUBLE_PRECISION "Use double precision" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# 精度设置
if(USE_DOUBLE_PRECISION)
    add_definitions(-DUSE_DOUBLE_PRECISION)
endif()

# 包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 源文件（仅包含已存在的文件）
set(SOURCES
    src/error_codes.cpp
    src/data_loader.cpp
    src/grid_structure.cpp
    src/cpu_interpolator.cpp
    src/api.cpp
)

set(CUDA_SOURCES
    src/cuda_interpolator.cu
    src/memory_manager.cu
)

# 静态库
add_library(point3d_interp STATIC ${SOURCES} ${CUDA_SOURCES})

# CUDA编译选项
target_compile_options(point3d_interp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
    >
)

# 链接CUDA库
target_link_libraries(point3d_interp PUBLIC cudart)

# 安装规则
install(TARGETS point3d_interp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/point3d_interp
    DESTINATION include
)

# 示例
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 测试
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
