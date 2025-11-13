# 示例程序CMakeLists.txt

# 基本使用示例
add_executable(basic_usage basic_usage.cpp)

# 链接库
target_link_libraries(basic_usage PRIVATE point3d_interp)

# 设置包含目录
target_include_directories(basic_usage PRIVATE ${PROJECT_SOURCE_DIR}/include)

# 安装
install(TARGETS basic_usage
    RUNTIME DESTINATION bin
)