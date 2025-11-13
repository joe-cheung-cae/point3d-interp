#include "point3d_interp/types.h"
#include <cuda_runtime.h>
#include <iostream>

namespace p3d {
namespace cuda {

/**
 * @brief CUDA error checking macro
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            return false; \
        } \
    } while (0)

/**
 * @brief GPU memory manager
 *
 * Responsible for CUDA memory allocation, deallocation and data transfer
 */
class GpuMemoryManager {
public:
    GpuMemoryManager() = default;
    ~GpuMemoryManager() { cleanup(); }

    // Disable copy
    GpuMemoryManager(const GpuMemoryManager&) = delete;
    GpuMemoryManager& operator=(const GpuMemoryManager&) = delete;

    // Allow move
    GpuMemoryManager(GpuMemoryManager&& other) noexcept;
    GpuMemoryManager& operator=(GpuMemoryManager&& other) noexcept;

    /**
     * @brief Allocate GPU memory
     * @param size Number of bytes
     * @return Whether successful
     */
    bool allocate(size_t size);

    /**
     * @brief Deallocate GPU memory
     */
    void deallocate();

    /**
     * @brief Copy data to GPU
     * @param host_data Host data pointer
     * @param size Number of bytes
     * @return Whether successful
     */
    bool copyToDevice(const void* host_data, size_t size);

    /**
     * @brief Copy data from GPU to host
     * @param host_data Host data pointer
     * @param size Number of bytes
     * @return Whether successful
     */
    bool copyToHost(void* host_data, size_t size);

    /**
     * @brief Get device pointer
     * @return Device memory pointer
     */
    void* getDevicePtr() const { return device_ptr_; }

    /**
     * @brief Get allocated memory size
     * @return Number of bytes
     */
    size_t getSize() const { return size_; }

    /**
     * @brief Check if memory is allocated
     * @return true if allocated
     */
    bool isAllocated() const { return device_ptr_ != nullptr; }

private:
    /**
     * @brief Clean up resources
     */
    void cleanup();

private:
    void* device_ptr_ = nullptr;  // Device memory pointer
    size_t size_ = 0;             // Allocated memory size
};

/**
 * @brief Templated GPU memory manager
 * Supports type-safe memory management
 */
template<typename T>
class GpuMemory {
public:
    GpuMemory() = default;
    ~GpuMemory() { deallocate(); }

    // Disable copy
    GpuMemory(const GpuMemory&) = delete;
    GpuMemory& operator=(const GpuMemory&) = delete;

    // Allow move
    GpuMemory(GpuMemory&& other) noexcept;
    GpuMemory& operator=(GpuMemory&& other) noexcept;

    /**
     * @brief Allocate GPU memory
     * @param count Number of elements
     * @return Whether successful
     */
    bool allocate(size_t count);

    /**
     * @brief Deallocate GPU memory
     */
    void deallocate();

    /**
     * @brief Copy data to GPU
     * @param host_data Host data pointer
     * @param count Number of elements
     * @return Whether successful
     */
    bool copyToDevice(const T* host_data, size_t count);

    /**
     * @brief Copy data from GPU to host
     * @param host_data Host data pointer
     * @param count Number of elements
     * @return Whether successful
     */
    bool copyToHost(T* host_data, size_t count);

    /**
     * @brief Get device pointer
     * @return Device memory pointer
     */
    T* getDevicePtr() const { return static_cast<T*>(device_ptr_); }

    /**
     * @brief Get number of allocated elements
     * @return Number of elements
     */
    size_t getCount() const { return size_ / sizeof(T); }

    /**
     * @brief Get allocated memory size
     * @return Number of bytes
     */
    size_t getSize() const { return size_; }

    /**
     * @brief Check if memory is allocated
     * @return true if allocated
     */
    bool isAllocated() const { return device_ptr_ != nullptr; }

private:
    void* device_ptr_ = nullptr;  // Device memory pointer
    size_t size_ = 0;             // Allocated memory size (bytes)
};

// Implementation of GpuMemoryManager

GpuMemoryManager::GpuMemoryManager(GpuMemoryManager&& other) noexcept
    : device_ptr_(other.device_ptr_), size_(other.size_)
{
    other.device_ptr_ = nullptr;
    other.size_ = 0;
}

GpuMemoryManager& GpuMemoryManager::operator=(GpuMemoryManager&& other) noexcept {
    if (this != &other) {
        cleanup();
        device_ptr_ = other.device_ptr_;
        size_ = other.size_;
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

bool GpuMemoryManager::allocate(size_t size) {
    if (isAllocated()) {
        deallocate();
    }

    CUDA_CHECK(cudaMalloc(&device_ptr_, size));
    size_ = size;
    return true;
}

void GpuMemoryManager::deallocate() {
    if (device_ptr_) {
        cudaFree(device_ptr_);
        device_ptr_ = nullptr;
        size_ = 0;
    }
}

bool GpuMemoryManager::copyToDevice(const void* host_data, size_t size) {
    if (!isAllocated() || size > size_) {
        return false;
    }

    CUDA_CHECK(cudaMemcpy(device_ptr_, host_data, size, cudaMemcpyHostToDevice));
    return true;
}

bool GpuMemoryManager::copyToHost(void* host_data, size_t size) {
    if (!isAllocated() || size > size_) {
        return false;
    }

    CUDA_CHECK(cudaMemcpy(host_data, device_ptr_, size, cudaMemcpyDeviceToHost));
    return true;
}

void GpuMemoryManager::cleanup() {
    deallocate();
}

// Implementation of GpuMemory<T>

template<typename T>
GpuMemory<T>::GpuMemory(GpuMemory<T>&& other) noexcept
    : device_ptr_(other.device_ptr_), size_(other.size_)
{
    other.device_ptr_ = nullptr;
    other.size_ = 0;
}

template<typename T>
GpuMemory<T>& GpuMemory<T>::operator=(GpuMemory<T>&& other) noexcept {
    if (this != &other) {
        deallocate();
        device_ptr_ = other.device_ptr_;
        size_ = other.size_;
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

template<typename T>
bool GpuMemory<T>::allocate(size_t count) {
    if (isAllocated()) {
        deallocate();
    }

    size_t size_bytes = count * sizeof(T);
    CUDA_CHECK(cudaMalloc(&device_ptr_, size_bytes));
    size_ = size_bytes;
    return true;
}

template<typename T>
void GpuMemory<T>::deallocate() {
    if (device_ptr_) {
        cudaFree(device_ptr_);
        device_ptr_ = nullptr;
        size_ = 0;
    }
}

template<typename T>
bool GpuMemory<T>::copyToDevice(const T* host_data, size_t count) {
    if (!isAllocated() || count * sizeof(T) > size_) {
        return false;
    }

    CUDA_CHECK(cudaMemcpy(device_ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    return true;
}

template<typename T>
bool GpuMemory<T>::copyToHost(T* host_data, size_t count) {
    if (!isAllocated() || count * sizeof(T) > size_) {
        return false;
    }

    CUDA_CHECK(cudaMemcpy(host_data, device_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    return true;
}

// Explicit instantiation of commonly used types
template class GpuMemory<Point3D>;
template class GpuMemory<MagneticFieldData>;
template class GpuMemory<InterpolationResult>;

} // namespace cuda
} // namespace p3d