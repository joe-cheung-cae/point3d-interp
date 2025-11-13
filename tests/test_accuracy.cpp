#include <gtest/gtest.h>
#include "point3d_interp/cpu_interpolator.h"
#include "point3d_interp/grid_structure.h"
#include <vector>
#include <cmath>

namespace p3d {
namespace test {

// 测试CPU和GPU插值结果的一致性
TEST(AccuracyTest, CPUInterpolationConsistency) {
    // 创建测试网格 (3x3x3)
    GridParams params;
    params.origin = Point3D(0.0f, 0.0f, 0.0f);
    params.spacing = Point3D(1.0f, 1.0f, 1.0f);
    params.dimensions = {3, 3, 3};
    params.update_bounds();

    RegularGrid3D grid(params);

    // 设置已知函数的测试数据: f(x,y,z) = x^2 + y^2 + z^2
    // 梯度: (2x, 2y, 2z)
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (size_t i = 0; i < field_data.size(); ++i) {
        const auto& coord = grid.getCoordinates()[i];
        float x = coord.x, y = coord.y, z = coord.z;
        field_data[i] = MagneticFieldData(
            x*x + y*y + z*z,  // B = x² + y² + z²
            2*x, 2*y, 2*z     // 梯度 = (2x, 2y, 2z)
        );
    }

    CPUInterpolator cpu_interp(grid);

    // 测试多个查询点
    std::vector<Point3D> test_points = {
        {0.5f, 0.5f, 0.5f},
        {1.2f, 0.8f, 1.5f},
        {0.3f, 1.7f, 0.9f},
        {2.1f, 1.3f, 0.7f}
    };

    for (const auto& point : test_points) {
        InterpolationResult result = cpu_interp.query(point);
        ASSERT_TRUE(result.valid);

        // 计算解析解
        float x = point.x, y = point.y, z = point.z;
        float expected_B = x*x + y*y + z*z;
        float expected_Bx = 2*x;
        float expected_By = 2*y;
        float expected_Bz = 2*z;

        // 检查插值精度 (三线性插值应该有很好的精度)
        EXPECT_NEAR(result.data.field_strength, expected_B, 1e-3f);
        EXPECT_NEAR(result.data.gradient_x, expected_Bx, 1e-3f);
        EXPECT_NEAR(result.data.gradient_y, expected_By, 1e-3f);
        EXPECT_NEAR(result.data.gradient_z, expected_Bz, 1e-3f);
    }
}

// 测试边界情况的插值精度
TEST(AccuracyTest, BoundaryInterpolation) {
    // 创建简单的2x2x2网格
    GridParams params;
    params.origin = Point3D(0.0f, 0.0f, 0.0f);
    params.spacing = Point3D(1.0f, 1.0f, 1.0f);
    params.dimensions = {2, 2, 2};
    params.update_bounds();

    RegularGrid3D grid(params);

    // 设置线性场: f(x,y,z) = x + 2*y + 3*z + 1
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (size_t i = 0; i < field_data.size(); ++i) {
        const auto& coord = grid.getCoordinates()[i];
        float x = coord.x, y = coord.y, z = coord.z;
        field_data[i] = MagneticFieldData(
            x + 2*y + 3*z + 1,  // B = x + 2y + 3z + 1
            1.0f, 2.0f, 3.0f   // 梯度 = (1, 2, 3)
        );
    }

    CPUInterpolator cpu_interp(grid);

    // 测试单元格内的多个点
    std::vector<Point3D> test_points;
    for (float x = 0.1f; x < 1.0f; x += 0.2f) {
        for (float y = 0.1f; y < 1.0f; y += 0.2f) {
            for (float z = 0.1f; z < 1.0f; z += 0.2f) {
                test_points.push_back(Point3D(x, y, z));
            }
        }
    }

    for (const auto& point : test_points) {
        InterpolationResult result = cpu_interp.query(point);
        ASSERT_TRUE(result.valid);

        // 计算解析解
        float expected_B = point.x + 2*point.y + 3*point.z + 1;

        // 对于线性场，三线性插值应该精确
        EXPECT_NEAR(result.data.field_strength, expected_B, 1e-6f);
        EXPECT_NEAR(result.data.gradient_x, 1.0f, 1e-6f);
        EXPECT_NEAR(result.data.gradient_y, 2.0f, 1e-6f);
        EXPECT_NEAR(result.data.gradient_z, 3.0f, 1e-6f);
    }
}

// 测试网格点插值的精确性
TEST(AccuracyTest, GridPointExactness) {
    // 创建测试网格
    GridParams params;
    params.origin = Point3D(0.0f, 0.0f, 0.0f);
    params.spacing = Point3D(0.5f, 0.5f, 0.5f);
    params.dimensions = {5, 5, 5};
    params.update_bounds();

    RegularGrid3D grid(params);

    // 设置随机数据
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (auto& data : field_data) {
        data = MagneticFieldData(
            static_cast<float>(rand()) / RAND_MAX,
            static_cast<float>(rand()) / RAND_MAX,
            static_cast<float>(rand()) / RAND_MAX,
            static_cast<float>(rand()) / RAND_MAX
        );
    }

    CPUInterpolator cpu_interp(grid);

    // 测试所有网格点
    const auto& coordinates = grid.getCoordinates();
    for (size_t i = 0; i < coordinates.size(); ++i) {
        const Point3D& point = coordinates[i];
        InterpolationResult result = cpu_interp.query(point);

        ASSERT_TRUE(result.valid);

        // 网格点插值应该精确匹配原始数据
        const MagneticFieldData& original = field_data[i];
        EXPECT_FLOAT_EQ(result.data.field_strength, original.field_strength);
        EXPECT_FLOAT_EQ(result.data.gradient_x, original.gradient_x);
        EXPECT_FLOAT_EQ(result.data.gradient_y, original.gradient_y);
        EXPECT_FLOAT_EQ(result.data.gradient_z, original.gradient_z);
    }
}

// 测试数值稳定性
TEST(AccuracyTest, NumericalStability) {
    // 创建大范围的网格
    GridParams params;
    params.origin = Point3D(-100.0f, -100.0f, -100.0f);
    params.spacing = Point3D(10.0f, 10.0f, 10.0f);
    params.dimensions = {21, 21, 21};  // 从-100到+100
    params.update_bounds();

    RegularGrid3D grid(params);

    // 设置简单的函数
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (size_t i = 0; i < field_data.size(); ++i) {
        const auto& coord = grid.getCoordinates()[i];
        field_data[i] = MagneticFieldData(
            coord.x + coord.y + coord.z,  // B = x + y + z
            1.0f, 1.0f, 1.0f
        );
    }

    CPUInterpolator cpu_interp(grid);

    // 测试远离原点的点
    std::vector<Point3D> test_points = {
        {50.0f, 25.0f, -75.0f},
        {-30.0f, 80.0f, 10.0f},
        {15.0f, -45.0f, 60.0f}
    };

    for (const auto& point : test_points) {
        InterpolationResult result = cpu_interp.query(point);
        ASSERT_TRUE(result.valid);

        // 计算解析解
        float expected_B = point.x + point.y + point.z;

        // 检查数值稳定性
        EXPECT_NEAR(result.data.field_strength, expected_B, 1e-2f);  // 放宽精度要求
        EXPECT_NEAR(result.data.gradient_x, 1.0f, 1e-2f);
        EXPECT_NEAR(result.data.gradient_y, 1.0f, 1e-2f);
        EXPECT_NEAR(result.data.gradient_z, 1.0f, 1e-2f);
    }
}

// 测试批量插值与单点插值的一致性
TEST(AccuracyTest, BatchVsSingleConsistency) {
    // 创建测试网格
    GridParams params;
    params.origin = Point3D(0.0f, 0.0f, 0.0f);
    params.spacing = Point3D(1.0f, 1.0f, 1.0f);
    params.dimensions = {4, 4, 4};
    params.update_bounds();

    RegularGrid3D grid(params);

    // 设置测试数据
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (auto& data : field_data) {
        data = MagneticFieldData(
            static_cast<float>(rand()) / RAND_MAX * 10.0f,
            static_cast<float>(rand()) / RAND_MAX * 2.0f,
            static_cast<float>(rand()) / RAND_MAX * 2.0f,
            static_cast<float>(rand()) / RAND_MAX * 2.0f
        );
    }

    CPUInterpolator cpu_interp(grid);

    // 生成测试点
    std::vector<Point3D> test_points;
    for (int i = 0; i < 50; ++i) {
        test_points.push_back(Point3D(
            static_cast<float>(rand()) / RAND_MAX * 3.0f,
            static_cast<float>(rand()) / RAND_MAX * 3.0f,
            static_cast<float>(rand()) / RAND_MAX * 3.0f
        ));
    }

    // 单点查询
    std::vector<InterpolationResult> single_results;
    for (const auto& point : test_points) {
        single_results.push_back(cpu_interp.query(point));
    }

    // 批量查询
    std::vector<InterpolationResult> batch_results(test_points.size());
    batch_results = cpu_interp.queryBatch(test_points);

    // 比较结果
    ASSERT_EQ(single_results.size(), batch_results.size());
    for (size_t i = 0; i < single_results.size(); ++i) {
        const auto& single = single_results[i];
        const auto& batch = batch_results[i];

        EXPECT_EQ(single.valid, batch.valid);

        if (single.valid && batch.valid) {
            EXPECT_FLOAT_EQ(single.data.field_strength, batch.data.field_strength);
            EXPECT_FLOAT_EQ(single.data.gradient_x, batch.data.gradient_x);
            EXPECT_FLOAT_EQ(single.data.gradient_y, batch.data.gradient_y);
            EXPECT_FLOAT_EQ(single.data.gradient_z, batch.data.gradient_z);
        }
    }
}

} // namespace test
} // namespace p3d