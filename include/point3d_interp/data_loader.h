#ifndef POINTER3D_INTERP_DATA_LOADER_H
#define POINTER3D_INTERP_DATA_LOADER_H

#include "types.h"
#include "error_codes.h"
#include <string>
#include <vector>
#include <memory>

namespace p3d {

/**
 * @brief CSV数据加载器
 *
 * 负责从CSV文件加载磁场数据，支持多种分隔符和格式
 */
class DataLoader {
public:
    DataLoader();
    ~DataLoader();

    /**
     * @brief 从CSV文件加载数据
     * @param filepath CSV文件路径
     * @param coordinates 输出坐标数组
     * @param field_data 输出磁场数据数组
     * @param grid_params 输出网格参数
     * @return 错误码
     */
    ErrorCode LoadFromCSV(
        const std::string& filepath,
        std::vector<Point3D>& coordinates,
        std::vector<MagneticFieldData>& field_data,
        GridParams& grid_params
    );

    /**
     * @brief 设置分隔符
     * @param delimiter 分隔符（默认逗号）
     */
    void SetDelimiter(char delimiter) { delimiter_ = delimiter; }

    /**
     * @brief 设置是否跳过标题行
     * @param skip_header 是否跳过标题行（默认true）
     */
    void SetSkipHeader(bool skip_header) { skip_header_ = skip_header; }

    /**
     * @brief 设置列索引
     * @param coord_cols 坐标列索引 [x, y, z]
     * @param field_cols 磁场数据列索引 [B, Bx, By, Bz]
     */
    void SetColumnIndices(
        const std::array<size_t, 3>& coord_cols,
        const std::array<size_t, 4>& field_cols
    );

private:
    /**
     * @brief 解析单行数据
     * @param line 输入行
     * @param point 输出坐标点
     * @param field 输出磁场数据
     * @return 是否解析成功
     */
    bool ParseLine(
        const std::string& line,
        Point3D& point,
        MagneticFieldData& field
    );

    /**
     * @brief 从数据中检测网格参数
     * @param coordinates 坐标数组
     * @param grid_params 输出网格参数
     * @return 是否检测成功
     */
    bool DetectGridParams(
        const std::vector<Point3D>& coordinates,
        GridParams& grid_params
    );

    /**
     * @brief 验证网格规则性
     * @param coordinates 坐标数组
     * @param grid_params 网格参数
     * @return 是否为规则网格
     */
    bool ValidateGridRegularity(
        const std::vector<Point3D>& coordinates,
        const GridParams& grid_params
    );

    /**
     * @brief 分割字符串
     * @param line 输入行
     * @param delimiter 分隔符
     * @return 分割后的字符串数组
     */
    std::vector<std::string> SplitString(
        const std::string& line,
        char delimiter
    );

    /**
     * @brief 转换字符串为数值
     * @param str 输入字符串
     * @param value 输出数值
     * @return 是否转换成功
     */
    template<typename T>
    bool StringToValue(const std::string& str, T& value);

private:
    char delimiter_;                    // 分隔符
    bool skip_header_;                  // 是否跳过标题行
    std::array<size_t, 3> coord_cols_;  // 坐标列索引
    std::array<size_t, 4> field_cols_;  // 磁场数据列索引
};

} // namespace p3d

#endif // POINTER3D_INTERP_DATA_LOADER_H