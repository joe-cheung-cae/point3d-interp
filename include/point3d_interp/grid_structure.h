#ifndef POINTER3D_INTERP_GRID_STRUCTURE_H
#define POINTER3D_INTERP_GRID_STRUCTURE_H

#include "types.h"
#include <vector>
#include <memory>

namespace p3d {

/**
 * @brief 规则网格结构类
 *
 * 管理三维规则网格的结构信息，提供坐标转换和索引计算功能
 */
class RegularGrid3D {
public:
    /**
     * @brief 构造函数
     * @param params 网格参数
     */
    explicit RegularGrid3D(const GridParams& params);

    /**
     * @brief 构造函数（从数据点自动构建）
     * @param coordinates 坐标数组
     * @param field_data 磁场数据数组
     */
    RegularGrid3D(
        const std::vector<Point3D>& coordinates,
        const std::vector<MagneticFieldData>& field_data
    );

    ~RegularGrid3D();

    // 禁止拷贝，允许移动
    RegularGrid3D(const RegularGrid3D&) = delete;
    RegularGrid3D& operator=(const RegularGrid3D&) = delete;
    RegularGrid3D(RegularGrid3D&&) noexcept;
    RegularGrid3D& operator=(RegularGrid3D&&) noexcept;

    /**
     * @brief 世界坐标转换为网格坐标
     * @param world_point 世界坐标点
     * @return 网格坐标
     */
    P3D_HOST_DEVICE
    Point3D worldToGrid(const Point3D& world_point) const;

    /**
     * @brief 网格坐标转换为世界坐标
     * @param grid_point 网格坐标点
     * @return 世界坐标
     */
    P3D_HOST_DEVICE
    Point3D gridToWorld(const Point3D& grid_point) const;

    /**
     * @brief 获取包含点的网格单元格的8个顶点索引
     * @param grid_coords 网格坐标
     * @param indices 输出8个顶点索引数组
     * @return 是否在有效范围内
     */
    P3D_HOST_DEVICE
    bool getCellVertexIndices(const Point3D& grid_coords, uint32_t indices[8]) const;

    /**
     * @brief 获取网格数据在数组中的索引
     * @param i x方向索引
     * @param j y方向索引
     * @param k z方向索引
     * @return 数组索引
     */
    P3D_HOST_DEVICE
    uint32_t getDataIndex(uint32_t i, uint32_t j, uint32_t k) const;

    /**
     * @brief 检查网格坐标是否在有效范围内
     * @param grid_coords 网格坐标
     * @return 是否有效
     */
    P3D_HOST_DEVICE
    bool isValidGridCoords(const Point3D& grid_coords) const;

    /**
     * @brief 获取网格参数
     * @return 网格参数
     */
    const GridParams& getParams() const { return params_; }

    /**
     * @brief 获取数据点数量
     * @return 数据点数量
     */
    size_t getDataCount() const;

    /**
     * @brief 获取所有坐标点
     * @return 坐标点数组
     */
    const std::vector<Point3D>& getCoordinates() const { return coordinates_; }

    /**
     * @brief 获取所有磁场数据
     * @return 磁场数据数组
     */
    const std::vector<MagneticFieldData>& getFieldData() const { return field_data_; }

private:
    /**
     * @brief 从坐标数据构建网格参数
     * @param coordinates 坐标数组
     */
    void buildFromCoordinates(const std::vector<Point3D>& coordinates);

    /**
     * @brief 验证网格数据的完整性
     * @return 是否有效
     */
    bool validateGridData() const;

private:
    GridParams params_;                           // 网格参数
    std::vector<Point3D> coordinates_;            // 坐标数据
    std::vector<MagneticFieldData> field_data_;   // 磁场数据
};

} // namespace p3d

#endif // POINTER3D_INTERP_GRID_STRUCTURE_H