#ifndef POINTER3D_INTERP_API_H
#define POINTER3D_INTERP_API_H

#include "types.h"
#include "error_codes.h"
#include <string>
#include <vector>
#include <memory>

namespace p3d {

/**
 * @brief 磁场数据插值器主类
 *
 * 这是库的主要接口类，负责数据加载、GPU资源管理和插值计算
 */
class MagneticFieldInterpolator {
public:
    /**
     * @brief 构造函数
     * @param use_gpu 是否使用GPU加速（默认true）
     * @param device_id GPU设备ID（默认0）
     */
    explicit MagneticFieldInterpolator(bool use_gpu = true, int device_id = 0);

    ~MagneticFieldInterpolator();

    // 禁止拷贝，允许移动
    MagneticFieldInterpolator(const MagneticFieldInterpolator&) = delete;
    MagneticFieldInterpolator& operator=(const MagneticFieldInterpolator&) = delete;
    MagneticFieldInterpolator(MagneticFieldInterpolator&&) noexcept;
    MagneticFieldInterpolator& operator=(MagneticFieldInterpolator&&) noexcept;

    /**
     * @brief 从CSV文件加载磁场数据
     * @param filepath CSV文件路径
     * @return 错误码
     */
    ErrorCode LoadFromCSV(const std::string& filepath);

    /**
     * @brief 从内存加载数据
     * @param points 坐标数组
     * @param field_data 磁场数据数组
     * @param count 数据点数量
     * @return 错误码
     */
    ErrorCode LoadFromMemory(
        const Point3D* points,
        const MagneticFieldData* field_data,
        size_t count
    );

    /**
     * @brief 单点插值查询
     * @param query_point 查询点坐标
     * @param result 输出结果
     * @return 错误码
     */
    ErrorCode Query(const Point3D& query_point, InterpolationResult& result);

    /**
     * @brief 批量插值查询
     * @param query_points 查询点数组
     * @param results 输出结果数组
     * @param count 查询点数量
     * @return 错误码
     */
    ErrorCode QueryBatch(
        const Point3D* query_points,
        InterpolationResult* results,
        size_t count
    );

    /**
     * @brief 获取网格参数
     * @return 网格参数
     */
    const GridParams& GetGridParams() const;

    /**
     * @brief 检查是否已加载数据
     * @return true表示已加载
     */
    bool IsDataLoaded() const;

    /**
     * @brief 获取数据点数量
     * @return 数据点数量
     */
    size_t GetDataPointCount() const;

private:
    /**
     * @brief 初始化GPU资源
     * @return 是否成功
     */
    bool InitializeGPU(int device_id);

    /**
     * @brief 释放GPU资源
     */
    void ReleaseGPU();

    /**
     * @brief 上传数据到GPU
     * @return 是否成功
     */
    bool UploadDataToGPU();

private:
    class Impl;  // Pimpl模式实现
    std::unique_ptr<Impl> impl_;
};

} // namespace p3d

#endif // POINTER3D_INTERP_API_H