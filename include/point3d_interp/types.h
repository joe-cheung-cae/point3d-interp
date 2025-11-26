#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>

namespace p3d {

// Precision selection: use float or double
#ifdef USE_DOUBLE_PRECISION
using Real = double;
#else
using Real = float;
#endif

// CUDA keyword definitions (only valid during CUDA compilation)
#ifdef __CUDACC__
    #define P3D_HOST_DEVICE __host__ __device__
#else
    #define P3D_HOST_DEVICE
#endif

// 3D point structure
struct Point3D {
    Real x, y, z;

    P3D_HOST_DEVICE
    Point3D(Real x_ = 0, Real y_ = 0, Real z_ = 0) : x(x_), y(y_), z(z_) {}

    P3D_HOST_DEVICE
    Point3D& operator+=(const Point3D& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    P3D_HOST_DEVICE
    Point3D operator+(const Point3D& other) const { return Point3D(x + other.x, y + other.y, z + other.z); }

    P3D_HOST_DEVICE
    Point3D operator-(const Point3D& other) const { return Point3D(x - other.x, y - other.y, z - other.z); }

    P3D_HOST_DEVICE
    Point3D operator*(Real scalar) const { return Point3D(x * scalar, y * scalar, z * scalar); }

    P3D_HOST_DEVICE
    Point3D operator/(Real scalar) const { return Point3D(x / scalar, y / scalar, z / scalar); }
};

// Magnetic field data structure
struct MagneticFieldData {
    Real Bx;  // Magnetic field component in x direction
    Real By;  // Magnetic field component in y direction
    Real Bz;  // Magnetic field component in z direction
    // Derivatives of Bx, By, Bz with respect to x, y, z
    Real dBx_dx, dBx_dy, dBx_dz;
    Real dBy_dx, dBy_dy, dBy_dz;
    Real dBz_dx, dBz_dy, dBz_dz;

    P3D_HOST_DEVICE
    MagneticFieldData(Real bx = 0, Real by = 0, Real bz = 0, Real dbx_dx = 0, Real dbx_dy = 0, Real dbx_dz = 0,
                      Real dby_dx = 0, Real dby_dy = 0, Real dby_dz = 0, Real dbz_dx = 0, Real dbz_dy = 0,
                      Real dbz_dz = 0)
        : Bx(bx),
          By(by),
          Bz(bz),
          dBx_dx(dbx_dx),
          dBx_dy(dbx_dy),
          dBx_dz(dbx_dz),
          dBy_dx(dby_dx),
          dBy_dy(dby_dy),
          dBy_dz(dby_dz),
          dBz_dx(dbz_dx),
          dBz_dy(dbz_dy),
          dBz_dz(dbz_dz) {}

    P3D_HOST_DEVICE
    MagneticFieldData& operator+=(const MagneticFieldData& other) {
        Bx += other.Bx;
        By += other.By;
        Bz += other.Bz;
        dBx_dx += other.dBx_dx;
        dBx_dy += other.dBx_dy;
        dBx_dz += other.dBx_dz;
        dBy_dx += other.dBy_dx;
        dBy_dy += other.dBy_dy;
        dBy_dz += other.dBy_dz;
        dBz_dx += other.dBz_dx;
        dBz_dy += other.dBz_dy;
        dBz_dz += other.dBz_dz;
        return *this;
    }

    P3D_HOST_DEVICE
    MagneticFieldData operator+(const MagneticFieldData& other) const {
        return MagneticFieldData(Bx + other.Bx, By + other.By, Bz + other.Bz, dBx_dx + other.dBx_dx,
                                 dBx_dy + other.dBx_dy, dBx_dz + other.dBx_dz, dBy_dx + other.dBy_dx,
                                 dBy_dy + other.dBy_dy, dBy_dz + other.dBy_dz, dBz_dx + other.dBz_dx,
                                 dBz_dy + other.dBz_dy, dBz_dz + other.dBz_dz);
    }

    P3D_HOST_DEVICE
    MagneticFieldData operator*(Real scalar) const {
        return MagneticFieldData(Bx * scalar, By * scalar, Bz * scalar, dBx_dx * scalar, dBx_dy * scalar,
                                 dBx_dz * scalar, dBy_dx * scalar, dBy_dy * scalar, dBy_dz * scalar, dBz_dx * scalar,
                                 dBz_dy * scalar, dBz_dz * scalar);
    }
};

// Grid parameters structure
struct GridParams {
    Point3D                 origin;      // Grid origin
    Point3D                 spacing;     // Grid spacing (dx, dy, dz)
    std::array<uint32_t, 3> dimensions;  // Grid dimensions (nx, ny, nz)

    // Boundaries
    Point3D min_bound;
    Point3D max_bound;

    GridParams() : origin(0, 0, 0), spacing(1, 1, 1), dimensions{0, 0, 0}, min_bound(0, 0, 0), max_bound(0, 0, 0) {}

    // Calculate boundaries
    void update_bounds() {
        min_bound   = origin;
        max_bound.x = origin.x + (dimensions[0] - 1) * spacing.x;
        max_bound.y = origin.y + (dimensions[1] - 1) * spacing.y;
        max_bound.z = origin.z + (dimensions[2] - 1) * spacing.z;
    }

    // Check if point is inside boundaries
    P3D_HOST_DEVICE
    bool is_point_inside(const Point3D& point) const {
        return (point.x >= min_bound.x && point.x <= max_bound.x) &&
               (point.y >= min_bound.y && point.y <= max_bound.y) && (point.z >= min_bound.z && point.z <= max_bound.z);
    }
};

// Interpolation result
struct InterpolationResult {
    MagneticFieldData data;
    bool              valid;  // Whether within valid range

    // Default constructor
    P3D_HOST_DEVICE
    InterpolationResult() : data(), valid(false) {}

    // Constructor with data and validity flag
    P3D_HOST_DEVICE
    InterpolationResult(const MagneticFieldData& d, bool v = true) : data(d), valid(v) {}

    // CUDA device-friendly initialization function
    P3D_HOST_DEVICE
    void initialize() {
        data.Bx     = 0;
        data.By     = 0;
        data.Bz     = 0;
        data.dBx_dx = 0;
        data.dBx_dy = 0;
        data.dBx_dz = 0;
        data.dBy_dx = 0;
        data.dBy_dy = 0;
        data.dBy_dz = 0;
        data.dBz_dx = 0;
        data.dBz_dy = 0;
        data.dBz_dz = 0;
        valid       = false;
    }

    P3D_HOST_DEVICE
    void set_result(const MagneticFieldData& d, bool v = true) {
        data  = d;
        valid = v;
    }
};

enum class InterpolationMethod { Trilinear, TricubicHermite, IDW };

enum class ExtrapolationMethod { None, NearestNeighbor, LinearExtrapolation };

// Spatial grid for GPU-accelerated neighbor finding
struct SpatialGrid {
    Point3D                 origin;        // Grid origin
    Point3D                 cell_size;     // Size of each cell (dx, dy, dz)
    std::array<uint32_t, 3> dimensions;    // Number of cells in each dimension (nx, ny, nz)
    std::vector<uint32_t>   cell_offsets;  // Offset array: cell_offsets[i] = start index of points in cell i
    std::vector<uint32_t>   cell_points;   // Point indices sorted by cell

    SpatialGrid() : origin(0, 0, 0), cell_size(1, 1, 1), dimensions{0, 0, 0} {}

    // Get total number of cells
    size_t get_num_cells() const { return dimensions[0] * dimensions[1] * dimensions[2]; }

    // Get cell index from 3D coordinates
    size_t get_cell_index(int ix, int iy, int iz) const {
        return ix + iy * dimensions[0] + iz * dimensions[0] * dimensions[1];
    }

    // Get 3D cell coordinates from world point
    void get_cell_coords(const Point3D& point, int& ix, int& iy, int& iz) const {
        ix = static_cast<int>((point.x - origin.x) / cell_size.x);
        iy = static_cast<int>((point.y - origin.y) / cell_size.y);
        iz = static_cast<int>((point.z - origin.z) / cell_size.z);

        // Clamp to bounds
        ix = std::max(0, std::min(ix, static_cast<int>(dimensions[0]) - 1));
        iy = std::max(0, std::min(iy, static_cast<int>(dimensions[1]) - 1));
        iz = std::max(0, std::min(iz, static_cast<int>(dimensions[2]) - 1));
    }
};

}  // namespace p3d
