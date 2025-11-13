# 测试CMakeLists.txt

# Google Test设置
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# 防止覆盖父项目的编译器/链接器设置
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 测试可执行文件
add_executable(unit_tests
    test_data_loader.cpp
    test_grid_structure.cpp
    test_cpu_interpolator.cpp
    test_api.cpp
    test_accuracy.cpp
    test_edge_cases.cpp
    benchmark.cpp
)

# 链接库
target_link_libraries(unit_tests
    PRIVATE
        point3d_interp
        GTest::gtest_main
)

# 设置C++标准
set_target_properties(unit_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 添加测试
add_test(NAME unit_tests COMMAND unit_tests)

# 基准测试
add_executable(performance_benchmark benchmark.cpp)
target_link_libraries(performance_benchmark PRIVATE point3d_interp)
set_target_properties(performance_benchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)