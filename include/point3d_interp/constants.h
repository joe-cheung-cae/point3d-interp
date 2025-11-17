#ifndef POINT3D_INTERP_CONSTANTS_H
#define POINT3D_INTERP_CONSTANTS_H

#include "types.h"

namespace p3d {

// ============================================================================
// Tolerance and Precision Constants
// ============================================================================

/**
 * @brief Default tolerance for floating point comparisons
 * Used in grid validation and coordinate comparisons
 */
constexpr Real DEFAULT_TOLERANCE = 1e-6f;

/**
 * @brief Small epsilon value for distance calculations
 * Used to avoid division by zero in interpolation algorithms
 */
constexpr Real DISTANCE_EPSILON = 1e-8f;

// ============================================================================
// Default Algorithm Parameters
// ============================================================================

/**
 * @brief Default IDW (Inverse Distance Weighting) power parameter
 * Controls the influence of distance on interpolation weights
 * Higher values give more weight to closer points
 */
constexpr Real DEFAULT_IDW_POWER = 2.0f;

/**
 * @brief Default maximum number of neighbors for unstructured interpolation
 * Set to 0 to use all available points
 */
constexpr size_t DEFAULT_MAX_NEIGHBORS = 0;

// ============================================================================
// CUDA Kernel Configuration
// ============================================================================

/**
 * @brief Standard CUDA block sizes for different kernel types
 * These values are optimized for common GPU architectures
 */

// Small block size for memory-intensive kernels
constexpr int CUDA_BLOCK_SIZE_256 = 256;

// Medium block size for balanced performance
constexpr int CUDA_BLOCK_SIZE_512 = 512;

// Large block size for compute-intensive kernels
constexpr int CUDA_BLOCK_SIZE_1024 = 1024;

// Default block size for general-purpose kernels
constexpr int CUDA_DEFAULT_BLOCK_SIZE = CUDA_BLOCK_SIZE_512;

// Minimum number of blocks to ensure proper GPU utilization
constexpr int CUDA_MIN_BLOCKS = 4;

// Maximum number of blocks to prevent excessive resource usage
constexpr int CUDA_MAX_BLOCKS = 1024;

// ============================================================================
// Algorithm-specific Constants
// ============================================================================

/**
 * @brief Maximum number of neighbors to consider in linear extrapolation
 * Used in LinearExtrapolate function for unstructured data
 */
constexpr size_t MAX_EXTRAPOLATION_NEIGHBORS = 5;

/**
 * @brief Shared memory limit factor for IDW kernel optimization
 * Limits shared memory usage to prevent kernel launch failures
 */
constexpr size_t SHARED_MEMORY_LIMIT_FACTOR = 4;

}  // namespace p3d

#endif  // POINT3D_INTERP_CONSTANTS_H