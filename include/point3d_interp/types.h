#ifndef POINTER3D_INTERP_TYPES_H
#define POINTER3D_INTERP_TYPES_H

#include <cstdint>
#include <array>

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
    Point3D(Real x_ = 0, Real y_ = 0, Real z_ = 0)
        : x(x_), y(y_), z(z_) {}

    P3D_HOST_DEVICE
    Point3D& operator+=(const Point3D& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    P3D_HOST_DEVICE
    Point3D operator+(const Point3D& other) const {
        return Point3D(x + other.x, y + other.y, z + other.z);
    }

    P3D_HOST_DEVICE
    Point3D operator-(const Point3D& other) const {
        return Point3D(x - other.x, y - other.y, z - other.z);
    }

    P3D_HOST_DEVICE
    Point3D operator*(Real scalar) const {
        return Point3D(x * scalar, y * scalar, z * scalar);
    }
};

// Magnetic field data structure
struct MagneticFieldData {
    Real field_strength;        // B
    Real gradient_x;            // Bx
    Real gradient_y;            // By
    Real gradient_z;            // Bz

    P3D_HOST_DEVICE
    MagneticFieldData(Real b = 0, Real bx = 0, Real by = 0, Real bz = 0)
        : field_strength(b), gradient_x(bx), gradient_y(by), gradient_z(bz) {}

    P3D_HOST_DEVICE
    MagneticFieldData& operator+=(const MagneticFieldData& other) {
        field_strength += other.field_strength;
        gradient_x += other.gradient_x;
        gradient_y += other.gradient_y;
        gradient_z += other.gradient_z;
        return *this;
    }

    P3D_HOST_DEVICE
    MagneticFieldData operator+(const MagneticFieldData& other) const {
        return MagneticFieldData(
            field_strength + other.field_strength,
            gradient_x + other.gradient_x,
            gradient_y + other.gradient_y,
            gradient_z + other.gradient_z
        );
    }

    P3D_HOST_DEVICE
    MagneticFieldData operator*(Real scalar) const {
        return MagneticFieldData(
            field_strength * scalar,
            gradient_x * scalar,
            gradient_y * scalar,
            gradient_z * scalar
        );
    }
};

// Grid parameters structure
struct GridParams {
    Point3D origin;             // Grid origin
    Point3D spacing;            // Grid spacing (dx, dy, dz)
    std::array<uint32_t, 3> dimensions;  // Grid dimensions (nx, ny, nz)

    // Boundaries
    Point3D min_bound;
    Point3D max_bound;

    GridParams()
        : origin(0, 0, 0), spacing(1, 1, 1),
          dimensions{0, 0, 0}, min_bound(0, 0, 0), max_bound(0, 0, 0) {}

    // Calculate boundaries
    void update_bounds() {
        min_bound = origin;
        max_bound.x = origin.x + (dimensions[0] - 1) * spacing.x;
        max_bound.y = origin.y + (dimensions[1] - 1) * spacing.y;
        max_bound.z = origin.z + (dimensions[2] - 1) * spacing.z;
    }

    // Check if point is inside boundaries
    P3D_HOST_DEVICE
    bool is_point_inside(const Point3D& point) const {
        return (point.x >= min_bound.x && point.x <= max_bound.x) &&
               (point.y >= min_bound.y && point.y <= max_bound.y) &&
               (point.z >= min_bound.z && point.z <= max_bound.z);
    }
};

// Interpolation result
struct InterpolationResult {
    MagneticFieldData data;
    bool valid;                 // Whether within valid range

    // Default constructor
    P3D_HOST_DEVICE
    InterpolationResult() : data(), valid(false) {}

    // Constructor with data and validity flag
    P3D_HOST_DEVICE
    InterpolationResult(const MagneticFieldData& d, bool v = true)
        : data(d), valid(v) {}

    // CUDA device-friendly initialization function
    P3D_HOST_DEVICE
    void initialize() {
        data.field_strength = 0;
        data.gradient_x = 0;
        data.gradient_y = 0;
        data.gradient_z = 0;
        valid = false;
    }

    P3D_HOST_DEVICE
    void set_result(const MagneticFieldData& d, bool v = true) {
        data = d;
        valid = v;
    }
};

} // namespace p3d

#endif // POINTER3D_INTERP_TYPES_H