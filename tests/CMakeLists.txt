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

# Test executable
add_executable(unit_tests
    test_data_loader.cpp
    test_grid_structure.cpp
    test_cpu_interpolator.cpp
    test_api.cpp
    test_accuracy.cpp
    test_edge_cases.cpp
    benchmark.cpp
)

# Link libraries
target_link_libraries(unit_tests
    PRIVATE
        point3d_interp
        GTest::gtest_main
)

# Set C++ standard
set_target_properties(unit_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add tests
add_test(NAME unit_tests COMMAND unit_tests)

# Benchmark
add_executable(performance_benchmark benchmark.cpp)
target_link_libraries(performance_benchmark PRIVATE point3d_interp)
set_target_properties(performance_benchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)