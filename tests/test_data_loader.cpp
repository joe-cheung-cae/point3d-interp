#include <gtest/gtest.h>
#include "point3d_interp/data_loader.h"
#include <fstream>
#include <string>

namespace p3d {
namespace test {

class DataLoaderTest : public ::testing::Test {
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
        test_file_path_ = "test_data.csv";

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

// 测试基本加载功能
TEST_F(DataLoaderTest, LoadValidCSV) {
    DataLoader                     loader;
    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;
    GridParams                     grid_params;

    ErrorCode result = loader.LoadFromCSV(test_file_path_, coordinates, field_data, grid_params);

    EXPECT_EQ(result, ErrorCode::Success);
    EXPECT_EQ(coordinates.size(), 8u);
    EXPECT_EQ(field_data.size(), 8u);

    // 检查网格参数
    EXPECT_EQ(grid_params.dimensions[0], 2u);
    EXPECT_EQ(grid_params.dimensions[1], 2u);
    EXPECT_EQ(grid_params.dimensions[2], 2u);

    EXPECT_FLOAT_EQ(grid_params.origin.x, 0.0f);
    EXPECT_FLOAT_EQ(grid_params.origin.y, 0.0f);
    EXPECT_FLOAT_EQ(grid_params.origin.z, 0.0f);

    EXPECT_FLOAT_EQ(grid_params.spacing.x, 1.0f);
    EXPECT_FLOAT_EQ(grid_params.spacing.y, 1.0f);
    EXPECT_FLOAT_EQ(grid_params.spacing.z, 1.0f);
}

// 测试数据内容正确性
TEST_F(DataLoaderTest, DataContentValidation) {
    DataLoader                     loader;
    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;
    GridParams                     grid_params;

    loader.LoadFromCSV(test_file_path_, coordinates, field_data, grid_params);

    // 检查第一个数据点
    EXPECT_FLOAT_EQ(coordinates[0].x, 0.0f);
    EXPECT_FLOAT_EQ(coordinates[0].y, 0.0f);
    EXPECT_FLOAT_EQ(coordinates[0].z, 0.0f);

    EXPECT_FLOAT_EQ(field_data[0].field_strength, 1.0f);
    EXPECT_FLOAT_EQ(field_data[0].gradient_x, 0.1f);
    EXPECT_FLOAT_EQ(field_data[0].gradient_y, 0.2f);
    EXPECT_FLOAT_EQ(field_data[0].gradient_z, 0.3f);

    // 检查最后一个数据点
    EXPECT_FLOAT_EQ(coordinates[7].x, 1.0f);
    EXPECT_FLOAT_EQ(coordinates[7].y, 1.0f);
    EXPECT_FLOAT_EQ(coordinates[7].z, 1.0f);

    EXPECT_FLOAT_EQ(field_data[7].field_strength, 1.2f);
    EXPECT_FLOAT_EQ(field_data[7].gradient_x, 0.15f);
    EXPECT_FLOAT_EQ(field_data[7].gradient_y, 0.16f);
    EXPECT_FLOAT_EQ(field_data[7].gradient_z, 0.35f);
}

// 测试文件不存在的情况
TEST_F(DataLoaderTest, FileNotFound) {
    DataLoader                     loader;
    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;
    GridParams                     grid_params;

    ErrorCode result = loader.LoadFromCSV("nonexistent_file.csv", coordinates, field_data, grid_params);

    EXPECT_EQ(result, ErrorCode::FileNotFound);
}

// 测试无效文件格式
TEST_F(DataLoaderTest, InvalidFileFormat) {
    // 创建无效的CSV文件
    std::string   invalid_file = "invalid.csv";
    std::ofstream file(invalid_file);
    file << "invalid,data,format\n";
    file << "1.0,2.0\n";  // 列数不匹配
    file.close();

    DataLoader                     loader;
    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;
    GridParams                     grid_params;

    ErrorCode result = loader.LoadFromCSV(invalid_file, coordinates, field_data, grid_params);

    EXPECT_EQ(result, ErrorCode::InvalidFileFormat);

    std::remove(invalid_file.c_str());
}

// 测试空文件
TEST_F(DataLoaderTest, EmptyFile) {
    std::string   empty_file = "empty.csv";
    std::ofstream file(empty_file);
    file << "x,y,z,B,Bx,By,Bz\n";  // 只有标题行
    file.close();

    DataLoader                     loader;
    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;
    GridParams                     grid_params;

    ErrorCode result = loader.LoadFromCSV(empty_file, coordinates, field_data, grid_params);

    EXPECT_EQ(result, ErrorCode::InvalidFileFormat);

    std::remove(empty_file.c_str());
}

// 测试列索引设置
TEST_F(DataLoaderTest, CustomColumnIndices) {
    DataLoader loader;

    // 设置自定义列索引 (假设数据顺序不同)
    loader.SetColumnIndices({1, 2, 0}, {4, 5, 6, 3});  // x,y,z,B,Bx,By,Bz -> y,z,x,Bx,By,Bz,B

    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;
    GridParams                     grid_params;

    ErrorCode result = loader.LoadFromCSV(test_file_path_, coordinates, field_data, grid_params);

    EXPECT_EQ(result, ErrorCode::Success);

    // 验证数据是否按自定义索引正确解析
    // 第一个数据行: 0.0,0.0,0.0,1.0,0.1,0.2,0.3
    // 按索引 {1,2,0} 解析坐标: y=0.0, z=0.0, x=0.0 -> (0.0, 0.0, 0.0)
    // 按索引 {4,5,6,3} 解析磁场: Bx=0.1, By=0.2, Bz=0.3, B=1.0
    EXPECT_FLOAT_EQ(coordinates[0].x, 0.0f);
    EXPECT_FLOAT_EQ(coordinates[0].y, 0.0f);
    EXPECT_FLOAT_EQ(coordinates[0].z, 0.0f);

    EXPECT_FLOAT_EQ(field_data[0].field_strength, 1.0f);
    EXPECT_FLOAT_EQ(field_data[0].gradient_x, 0.1f);
    EXPECT_FLOAT_EQ(field_data[0].gradient_y, 0.2f);
    EXPECT_FLOAT_EQ(field_data[0].gradient_z, 0.3f);
}

}  // namespace test
}  // namespace p3d