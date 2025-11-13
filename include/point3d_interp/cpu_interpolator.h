#ifndef POINTER3D_INTERP_CPU_INTERPOLATOR_H
#define POINTER3D_INTERP_CPU_INTERPOLATOR_H

#include "types.h"
#include "grid_structure.h"
#include <vector>

namespace p3d {

/**
 * @brief CPU端三线性插值器
 *
 * 提供CPU版本的三线性插值实现，用于验证GPU版本的正确性
 */
class CPUInterpolator {
public:
    /**
     * @brief 构造函数
     * @param grid 规则网格对象
     */
    explicit CPUInterpolator(const RegularGrid3D& grid);

    ~CPUInterpolator();

    // 禁止拷贝，允许移动
    CPUInterpolator(const CPUInterpolator&) = delete;
    CPUInterpolator& operator=(const CPUInterpolator&) = delete;
    CPUInterpolator(CPUInterpolator&&) noexcept;
    CPUInterpolator& operator=(CPUInterpolator&&) noexcept;

    /**
     * @brief 单点插值查询
     * @param query_point 查询点坐标
     * @return 插值结果
     */
    InterpolationResult query(const Point3D& query_point) const;

    /**
     * @brief 批量插值查询
     * @param query_points 查询点数组
     * @return 插值结果数组
     */
    std::vector<InterpolationResult> queryBatch(
        const std::vector<Point3D>& query_points
    ) const;

    /**
     * @brief 获取网格引用
     * @return 网格对象的常量引用
     */
    const RegularGrid3D& getGrid() const { return grid_; }

private:
    /**
     * @brief 执行三线性插值计算
     * @param grid_coords 网格坐标
     * @param vertex_data 8个顶点的磁场数据
     * @param tx x方向局部坐标 (0-1)
     * @param ty y方向局部坐标 (0-1)
     * @param tz z方向局部坐标 (0-1)
     * @return 插值结果
     */
    MagneticFieldData trilinearInterpolate(
        const MagneticFieldData vertex_data[8],
        Real tx, Real ty, Real tz
    ) const;

    /**
     * @brief 获取单元格顶点数据
     * @param indices 8个顶点索引
     * @param vertex_data 输出顶点数据数组
     */
    void getVertexData(
        const uint32_t indices[8],
        MagneticFieldData vertex_data[8]
    ) const;

private:
    const RegularGrid3D& grid_;  // 网格引用
};

} // namespace p3d

#endif // POINTER3D_INTERP_CPU_INTERPOLATOR_H