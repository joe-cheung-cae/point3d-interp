#include <gtest/gtest.h>
#include "point3d_interp/grid_structure.h"
#include <vector>

namespace p3d {
namespace test {

// 测试网格参数构造函数
TEST(GridStructureTest, GridParamsConstructor) {
    GridParams params;
    EXPECT_EQ(params.dimensions[0], 0u);
    EXPECT_EQ(params.dimensions[1], 0u);
    EXPECT_EQ(params.dimensions[2], 0u);

    EXPECT_FLOAT_EQ(params.origin.x, 0.0f);
    EXPECT_FLOAT_EQ(params.origin.y, 0.0f);
    EXPECT_FLOAT_EQ(params.origin.z, 0.0f);

    EXPECT_FLOAT_EQ(params.spacing.x, 1.0f);
    EXPECT_FLOAT_EQ(params.spacing.y, 1.0f);
    EXPECT_FLOAT_EQ(params.spacing.z, 1.0f);
}

// 测试网格参数边界更新
TEST(GridStructureTest, GridParamsUpdateBounds) {
    GridParams params;
    params.origin     = Point3D(1.0f, 2.0f, 3.0f);
    params.spacing    = Point3D(0.5f, 1.0f, 2.0f);
    params.dimensions = {10, 5, 3};

    params.update_bounds();

    EXPECT_FLOAT_EQ(params.min_bound.x, 1.0f);
    EXPECT_FLOAT_EQ(params.min_bound.y, 2.0f);
    EXPECT_FLOAT_EQ(params.min_bound.z, 3.0f);

    EXPECT_FLOAT_EQ(params.max_bound.x, 1.0f + 9 * 0.5f);  // 1.0 + 4.5 = 5.5
    EXPECT_FLOAT_EQ(params.max_bound.y, 2.0f + 4 * 1.0f);  // 2.0 + 4.0 = 6.0
    EXPECT_FLOAT_EQ(params.max_bound.z, 3.0f + 2 * 2.0f);  // 3.0 + 4.0 = 7.0
}

// 测试点是否在边界内
TEST(GridStructureTest, PointInsideBounds) {
    GridParams params;
    params.min_bound = Point3D(0.0f, 0.0f, 0.0f);
    params.max_bound = Point3D(10.0f, 10.0f, 10.0f);

    // 测试边界内的点
    EXPECT_TRUE(params.is_point_inside(Point3D(5.0f, 5.0f, 5.0f)));
    EXPECT_TRUE(params.is_point_inside(Point3D(0.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(params.is_point_inside(Point3D(10.0f, 10.0f, 10.0f)));

    // 测试边界外的点
    EXPECT_FALSE(params.is_point_inside(Point3D(-1.0f, 5.0f, 5.0f)));
    EXPECT_FALSE(params.is_point_inside(Point3D(11.0f, 5.0f, 5.0f)));
    EXPECT_FALSE(params.is_point_inside(Point3D(5.0f, -1.0f, 5.0f)));
    EXPECT_FALSE(params.is_point_inside(Point3D(5.0f, 11.0f, 5.0f)));
    EXPECT_FALSE(params.is_point_inside(Point3D(5.0f, 5.0f, -1.0f)));
    EXPECT_FALSE(params.is_point_inside(Point3D(5.0f, 5.0f, 11.0f)));
}

// 测试RegularGrid3D构造函数（参数版本）
TEST(GridStructureTest, RegularGridConstructorWithParams) {
    GridParams params;
    params.origin     = Point3D(0.0f, 0.0f, 0.0f);
    params.spacing    = Point3D(1.0f, 1.0f, 1.0f);
    params.dimensions = {2, 2, 2};
    params.update_bounds();

    RegularGrid3D grid(params);

    EXPECT_EQ(grid.getDataCount(), 8u);  // 2x2x2 = 8

    const auto& grid_params = grid.getParams();
    EXPECT_EQ(grid_params.dimensions[0], 2u);
    EXPECT_EQ(grid_params.dimensions[1], 2u);
    EXPECT_EQ(grid_params.dimensions[2], 2u);

    // 检查生成的坐标点
    const auto& coordinates = grid.getCoordinates();
    EXPECT_EQ(coordinates.size(), 8u);

    // 检查第一个点 (0,0,0)
    EXPECT_FLOAT_EQ(coordinates[0].x, 0.0f);
    EXPECT_FLOAT_EQ(coordinates[0].y, 0.0f);
    EXPECT_FLOAT_EQ(coordinates[0].z, 0.0f);

    // 检查最后一个点 (1,1,1)
    EXPECT_FLOAT_EQ(coordinates[7].x, 1.0f);
    EXPECT_FLOAT_EQ(coordinates[7].y, 1.0f);
    EXPECT_FLOAT_EQ(coordinates[7].z, 1.0f);
}

// 测试RegularGrid3D构造函数（数据版本）
TEST(GridStructureTest, RegularGridConstructorWithData) {
    std::vector<Point3D> coordinates = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f},
                                        {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}};

    std::vector<MagneticFieldData> field_data(8, MagneticFieldData(1.0f, 0.1f, 0.2f, 0.3f));

    RegularGrid3D grid(coordinates, field_data);

    EXPECT_EQ(grid.getDataCount(), 8u);

    const auto& params = grid.getParams();
    EXPECT_EQ(params.dimensions[0], 2u);
    EXPECT_EQ(params.dimensions[1], 2u);
    EXPECT_EQ(params.dimensions[2], 2u);

    EXPECT_FLOAT_EQ(params.origin.x, 0.0f);
    EXPECT_FLOAT_EQ(params.origin.y, 0.0f);
    EXPECT_FLOAT_EQ(params.origin.z, 0.0f);

    EXPECT_FLOAT_EQ(params.spacing.x, 1.0f);
    EXPECT_FLOAT_EQ(params.spacing.y, 1.0f);
    EXPECT_FLOAT_EQ(params.spacing.z, 1.0f);
}

// 测试坐标转换
TEST(GridStructureTest, CoordinateConversion) {
    GridParams params;
    params.origin     = Point3D(1.0f, 2.0f, 3.0f);
    params.spacing    = Point3D(0.5f, 1.0f, 2.0f);
    params.dimensions = {10, 5, 3};
    params.update_bounds();

    RegularGrid3D grid(params);

    // 测试世界坐标到网格坐标的转换
    Point3D world_point(2.0f, 4.0f, 7.0f);  // 世界坐标
    Point3D grid_coords = grid.worldToGrid(world_point);

    // 期望的网格坐标: (2.0-1.0)/0.5 = 2.0, (4.0-2.0)/1.0 = 2.0, (7.0-3.0)/2.0 = 2.0
    EXPECT_FLOAT_EQ(grid_coords.x, 2.0f);
    EXPECT_FLOAT_EQ(grid_coords.y, 2.0f);
    EXPECT_FLOAT_EQ(grid_coords.z, 2.0f);

    // 测试网格坐标到世界坐标的转换
    Point3D world_coords = grid.gridToWorld(grid_coords);

    EXPECT_FLOAT_EQ(world_coords.x, 2.0f);
    EXPECT_FLOAT_EQ(world_coords.y, 4.0f);
    EXPECT_FLOAT_EQ(world_coords.z, 7.0f);
}

// 测试网格坐标有效性检查
TEST(GridStructureTest, ValidGridCoordinates) {
    GridParams params;
    params.dimensions = {5, 5, 5};

    RegularGrid3D grid(params);

    // 有效的网格坐标
    EXPECT_TRUE(grid.isValidGridCoords(Point3D(0.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(grid.isValidGridCoords(Point3D(2.5f, 2.5f, 2.5f)));
    EXPECT_TRUE(grid.isValidGridCoords(Point3D(3.9f, 3.9f, 3.9f)));

    // 无效的网格坐标（超出插值范围）
    EXPECT_FALSE(grid.isValidGridCoords(Point3D(-0.1f, 2.5f, 2.5f)));
    EXPECT_FALSE(grid.isValidGridCoords(Point3D(4.1f, 2.5f, 2.5f)));
    EXPECT_FALSE(grid.isValidGridCoords(Point3D(2.5f, -0.1f, 2.5f)));
    EXPECT_FALSE(grid.isValidGridCoords(Point3D(2.5f, 4.1f, 2.5f)));
    EXPECT_FALSE(grid.isValidGridCoords(Point3D(2.5f, 2.5f, -0.1f)));
    EXPECT_FALSE(grid.isValidGridCoords(Point3D(2.5f, 2.5f, 4.1f)));
}

// 测试单元格顶点索引获取
TEST(GridStructureTest, CellVertexIndices) {
    GridParams params;
    params.dimensions = {3, 3, 3};

    RegularGrid3D grid(params);

    // 测试网格坐标 (1.3, 1.7, 0.5) 应该在单元格 (1,1,0) 内
    Point3D  grid_coords(1.3f, 1.7f, 0.5f);
    uint32_t indices[8];

    bool success = grid.getCellVertexIndices(grid_coords, indices);
    EXPECT_TRUE(success);

    // 单元格 (1,1,0) 的8个顶点索引
    // 对应的数据索引应该是:
    // (1,1,0), (2,1,0), (1,2,0), (2,2,0),
    // (1,1,1), (2,1,1), (1,2,1), (2,2,1)

    EXPECT_EQ(indices[0], grid.getDataIndex(1, 1, 0));  // (1,1,0)
    EXPECT_EQ(indices[1], grid.getDataIndex(2, 1, 0));  // (2,1,0)
    EXPECT_EQ(indices[2], grid.getDataIndex(1, 2, 0));  // (1,2,0)
    EXPECT_EQ(indices[3], grid.getDataIndex(2, 2, 0));  // (2,2,0)
    EXPECT_EQ(indices[4], grid.getDataIndex(1, 1, 1));  // (1,1,1)
    EXPECT_EQ(indices[5], grid.getDataIndex(2, 1, 1));  // (2,1,1)
    EXPECT_EQ(indices[6], grid.getDataIndex(1, 2, 1));  // (1,2,1)
    EXPECT_EQ(indices[7], grid.getDataIndex(2, 2, 1));  // (2,2,1)
}

// 测试数据索引计算
TEST(GridStructureTest, DataIndexCalculation) {
    GridParams params;
    params.dimensions = {4, 3, 2};  // 4x3x2 = 24 个点

    RegularGrid3D grid(params);

    // 测试边界情况
    EXPECT_EQ(grid.getDataIndex(0, 0, 0), 0u);
    EXPECT_EQ(grid.getDataIndex(3, 0, 0), 3u);
    EXPECT_EQ(grid.getDataIndex(0, 2, 0), 8u);   // 2 * 4
    EXPECT_EQ(grid.getDataIndex(0, 0, 1), 12u);  // 1 * 4 * 3
    EXPECT_EQ(grid.getDataIndex(3, 2, 1), 23u);  // 3 + 2*4 + 1*12
}

// 测试无效网格数据
TEST(GridStructureTest, InvalidGridData) {
    // 空坐标数据
    std::vector<Point3D>           empty_coords;
    std::vector<MagneticFieldData> empty_data;

    EXPECT_THROW({ RegularGrid3D grid(empty_coords, empty_data); }, std::invalid_argument);

    // 坐标和数据大小不匹配
    std::vector<Point3D>           coords = {{0, 0, 0}, {1, 0, 0}};
    std::vector<MagneticFieldData> data   = {MagneticFieldData()};  // 只有1个数据点

    EXPECT_THROW({ RegularGrid3D grid(coords, data); }, std::invalid_argument);

    // 不规则网格数据
    std::vector<Point3D> irregular_coords = {
        {0, 0, 0}, {1, 0, 0},   {2, 0, 0},  // x方向正常
        {0, 1, 0}, {1, 0.5, 0}, {2, 1, 0}   // y方向不均匀
    };
    std::vector<MagneticFieldData> irregular_data(6, MagneticFieldData());

    EXPECT_THROW({ RegularGrid3D grid(irregular_coords, irregular_data); }, std::invalid_argument);
}

}  // namespace test
}  // namespace p3d