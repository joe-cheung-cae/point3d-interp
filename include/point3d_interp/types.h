#ifndef POINTER3D_INTERP_TYPES_H
#define POINTER3D_INTERP_TYPES_H

#include <cstdint>
#include <array>

namespace p3d {

// 精度选择：使用float或double
#ifdef USE_DOUBLE_PRECISION
using Real = double;
#else
using Real = float;
#endif

// CUDA 关键字定义（仅在CUDA编译时有效）
#ifdef __CUDACC__
#define P3D_HOST_DEVICE __host__ __device__
#else
#define P3D_HOST_DEVICE
#endif

// 三维点结构
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

// 磁场数据结构
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

// 网格参数结构
struct GridParams {
    Point3D origin;             // 网格起点
    Point3D spacing;            // 网格间距 (dx, dy, dz)
    std::array<uint32_t, 3> dimensions;  // 网格维度 (nx, ny, nz)

    // 边界
    Point3D min_bound;
    Point3D max_bound;

    GridParams()
        : origin(0, 0, 0), spacing(1, 1, 1),
          dimensions{0, 0, 0}, min_bound(0, 0, 0), max_bound(0, 0, 0) {}

    // 计算边界
    void update_bounds() {
        min_bound = origin;
        max_bound.x = origin.x + (dimensions[0] - 1) * spacing.x;
        max_bound.y = origin.y + (dimensions[1] - 1) * spacing.y;
        max_bound.z = origin.z + (dimensions[2] - 1) * spacing.z;
    }

    // 检查点是否在边界内
    P3D_HOST_DEVICE
    bool is_point_inside(const Point3D& point) const {
        return (point.x >= min_bound.x && point.x <= max_bound.x) &&
               (point.y >= min_bound.y && point.y <= max_bound.y) &&
               (point.z >= min_bound.z && point.z <= max_bound.z);
    }
};

// 插值结果
struct InterpolationResult {
    MagneticFieldData data;
    bool valid;                 // 是否在有效范围内

    // 默认构造函数
    P3D_HOST_DEVICE
    InterpolationResult() : data(), valid(false) {}

    // 带参数构造函数
    P3D_HOST_DEVICE
    InterpolationResult(const MagneticFieldData& d, bool v = true)
        : data(d), valid(v) {}

    // CUDA设备友好的初始化函数
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