# Example Program CMakeLists.txt

# Basic Usage Example
add_executable(basic_usage basic_usage.cpp)

# Link libraries
target_link_libraries(basic_usage PRIVATE point3d_interp)

# Set include directories
target_include_directories(basic_usage PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Install
install(TARGETS basic_usage
    RUNTIME DESTINATION bin
)