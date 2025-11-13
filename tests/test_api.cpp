#include <gtest/gtest.h>
#include "point3d_interp/api.h"
#include <fstream>
#include <string>

namespace p3d {
namespace test {

class APITest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建临时测试文件
        CreateTestCSVFile();
    }

    void TearDown() override {
        // 清理临时文件
        std::remove(test_file_path_.c_str());
    }

    void CreateTestCSVFile() {
        test_file_path_ = "api_test.csv";

        std::ofstream file(test_file_path_);
        ASSERT_TRUE(file.is_open());

        // 写入测试数据 (2x2x2网格)
        file << "x,y,z,B,Bx,By,Bz\n";
        file << "0.0,0.0,0.0,1.0,0.1,0.2,0.3\n";
        file << "1.0,0.0,0.0,1.1,0.15,0.18,0.32\n";
        file << "0.0,1.0,0.0,0.9,0.08,0.22,0.28\n";
        file << "1.0,1.0,0.0,1.0,0.13,0.19,0.31\n";
        file << "0.0,0.0,1.0,1.2,0.12,0.18,0.35\n";
        file << "1.0,0.0,1.0,1.3,0.17,0.14,0.37\n";
        file << "0.0,1.0,1.0,1.1,0.1,0.2,0.33\n";
        file << "1.0,1.0,1.0,1.2,0.15,0.16,0.35\n";

        file.close();
    }

    std::string test_file_path_;
};

// 测试基本API功能
TEST_F(APITest, BasicAPIUsage) {
    MagneticFieldInterpolator interp;

    // 加载数据
    ErrorCode err = interp.LoadFromCSV(test_file_path_);
    EXPECT_EQ(err, ErrorCode::Success);

    // 检查数据加载状态
    EXPECT_TRUE(interp.IsDataLoaded());
    EXPECT_EQ(interp.GetDataPointCount(), 8u);

    // 检查网格参数
    const auto& params = interp.GetGridParams();
    EXPECT_EQ(params.dimensions[0], 2u);
    EXPECT_EQ(params.dimensions[1], 2u);
    EXPECT_EQ(params.dimensions[2], 2u);
}

// 测试单点查询
TEST_F(APITest, SinglePointQuery) {
    MagneticFieldInterpolator interp;
    interp.LoadFromCSV(test_file_path_);

    Point3D query_point(0.5f, 0.5f, 0.5f);
    InterpolationResult result;

    ErrorCode err = interp.Query(query_point, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(result.valid);

    // 检查结果合理性 (应该在1.0到4.0之间)
    EXPECT_GE(result.data.field_strength, 1.0f);
    EXPECT_LE(result.data.field_strength, 4.0f);
}

// 测试批量查询
TEST_F(APITest, BatchQuery) {
    MagneticFieldInterpolator interp;
    interp.LoadFromCSV(test_file_path_);

    std::vector<Point3D> query_points = {
        {0.0f, 0.0f, 0.0f},
        {0.5f, 0.5f, 0.5f},
        {1.0f, 1.0f, 1.0f}
    };

    std::vector<InterpolationResult> results(query_points.size());

    ErrorCode err = interp.QueryBatch(query_points.data(), results.data(), query_points.size());
    EXPECT_EQ(err, ErrorCode::Success);

    // 检查所有结果都有效
    for (const auto& result : results) {
        EXPECT_TRUE(result.valid);
    }
}

// 测试边界外查询
TEST_F(APITest, OutOfBoundsQuery) {
    MagneticFieldInterpolator interp;
    interp.LoadFromCSV(test_file_path_);

    Point3D query_point(-1.0f, 0.5f, 0.5f);  // 边界外
    InterpolationResult result;

    ErrorCode err = interp.Query(query_point, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_FALSE(result.valid);
}

// 测试从内存加载数据
TEST_F(APITest, LoadFromMemory) {
    // 准备测试数据
    std::vector<Point3D> coordinates = {
        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}
    };

    std::vector<MagneticFieldData> field_data(coordinates.size(),
        MagneticFieldData(1.0f, 0.1f, 0.2f, 0.3f));

    MagneticFieldInterpolator interp;

    ErrorCode err = interp.LoadFromMemory(
        coordinates.data(),
        field_data.data(),
        coordinates.size()
    );

    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(interp.IsDataLoaded());
    EXPECT_EQ(interp.GetDataPointCount(), 8u);
}

// 测试文件不存在的情况
TEST_F(APITest, FileNotFound) {
    MagneticFieldInterpolator interp;

    ErrorCode err = interp.LoadFromCSV("nonexistent_file.csv");
    EXPECT_EQ(err, ErrorCode::FileNotFound);
    EXPECT_FALSE(interp.IsDataLoaded());
}

// 测试空查询
TEST_F(APITest, EmptyBatchQuery) {
    MagneticFieldInterpolator interp;
    interp.LoadFromCSV(test_file_path_);

    ErrorCode err = interp.QueryBatch(nullptr, nullptr, 0);
    EXPECT_EQ(err, ErrorCode::InvalidParameter);
}

// 测试未加载数据时的查询
TEST(APITest, QueryWithoutData) {
    MagneticFieldInterpolator interp;  // 未加载数据

    Point3D query_point(0.0f, 0.0f, 0.0f);
    InterpolationResult result;

    ErrorCode err = interp.Query(query_point, result);
    EXPECT_EQ(err, ErrorCode::DataNotLoaded);
}

// 测试移动语义
TEST_F(APITest, MoveSemantics) {
    MagneticFieldInterpolator interp1;
    interp1.LoadFromCSV(test_file_path_);

    // 测试移动构造函数
    MagneticFieldInterpolator interp2(std::move(interp1));
    EXPECT_TRUE(interp2.IsDataLoaded());
    EXPECT_FALSE(interp1.IsDataLoaded());  // 移动后源对象无效

    // 重新加载数据到interp1
    interp1.LoadFromCSV(test_file_path_);

    // 测试移动赋值
    MagneticFieldInterpolator interp3;
    interp3 = std::move(interp1);
    EXPECT_TRUE(interp3.IsDataLoaded());
    EXPECT_FALSE(interp1.IsDataLoaded());
}

} // namespace test
} // namespace p3d