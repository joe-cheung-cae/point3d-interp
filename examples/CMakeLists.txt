# Example Program CMakeLists.txt

# Basic Usage Example
add_executable(basic_usage basic_usage.cpp)

# Direct CUDA Kernel Example
add_executable(cuda_kernel_direct cuda_kernel_direct.cu)

# Link libraries
target_link_libraries(basic_usage PRIVATE point3d_interp)
target_link_libraries(cuda_kernel_direct PRIVATE point3d_interp)

# Set include directories
target_include_directories(basic_usage PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(cuda_kernel_direct PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Install
install(TARGETS basic_usage cuda_kernel_direct
    RUNTIME DESTINATION bin
)