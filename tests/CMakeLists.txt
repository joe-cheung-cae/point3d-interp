# Test CMakeLists.txt

# Google Test setup
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Data-related tests
add_executable(test_data
    test_data_loader.cpp
    test_grid_structure.cpp
)

target_link_libraries(test_data
    PRIVATE
    point3d_interp
    GTest::gtest_main
)

set_target_properties(test_data PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_test(NAME test_data COMMAND test_data)

# Interpolation tests
add_executable(test_interpolation
    test_cpu_interpolator.cpp
    test_accuracy.cpp
)

target_link_libraries(test_interpolation
    PRIVATE
    point3d_interp
    GTest::gtest_main
)

set_target_properties(test_interpolation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_test(NAME test_interpolation COMMAND test_interpolation)

# API tests
add_executable(test_api
    test_api.cpp
)

target_link_libraries(test_api
    PRIVATE
    point3d_interp
    GTest::gtest_main
)

set_target_properties(test_api PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_test(NAME test_api COMMAND test_api)

# Edge cases tests
add_executable(test_edge_cases
    test_edge_cases.cpp
)

target_link_libraries(test_edge_cases
    PRIVATE
    point3d_interp
    GTest::gtest_main
)

set_target_properties(test_edge_cases PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_test(NAME test_edge_cases COMMAND test_edge_cases)

# Benchmark
add_executable(performance_benchmark benchmark.cpp)
target_link_libraries(performance_benchmark PRIVATE point3d_interp)
set_target_properties(performance_benchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Split benchmarks
add_executable(benchmark_10x10x10 benchmark_10x10x10.cpp)
target_link_libraries(benchmark_10x10x10 PRIVATE point3d_interp)
set_target_properties(benchmark_10x10x10 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_executable(benchmark_20x20x20 benchmark_20x20x20.cpp)
target_link_libraries(benchmark_20x20x20 PRIVATE point3d_interp)
set_target_properties(benchmark_20x20x20 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_executable(benchmark_30x30x30 benchmark_30x30x30.cpp)
target_link_libraries(benchmark_30x30x30 PRIVATE point3d_interp)
set_target_properties(benchmark_30x30x30 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_executable(benchmark_50x50x50 benchmark_50x50x50.cpp)
target_link_libraries(benchmark_50x50x50 PRIVATE point3d_interp)
set_target_properties(benchmark_50x50x50 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_executable(benchmark_memory benchmark_memory.cpp)
target_link_libraries(benchmark_memory PRIVATE point3d_interp)
set_target_properties(benchmark_memory PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)