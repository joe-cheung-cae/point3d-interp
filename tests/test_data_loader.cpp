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

        // 写入测试数据 (2x2x2网格) 包含导数
        file << "x,y,z,Bx,By,Bz,dBx_dx,dBx_dy,dBx_dz,dBy_dx,dBy_dy,dBy_dz,dBz_dx,dBz_dy,dBz_dz\n";
        file << "0.0,0.0,0.0,0.1,0.2,0.3,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09\n";
        file << "1.0,0.0,0.0,0.15,0.18,0.32,0.011,0.021,0.031,0.041,0.051,0.061,0.071,0.081,0.091\n";
        file << "0.0,1.0,0.0,0.08,0.22,0.28,0.012,0.022,0.032,0.042,0.052,0.062,0.072,0.082,0.092\n";
        file << "1.0,1.0,0.0,0.13,0.19,0.31,0.013,0.023,0.033,0.043,0.053,0.063,0.073,0.083,0.093\n";
        file << "0.0,0.0,1.0,0.12,0.18,0.35,0.014,0.024,0.034,0.044,0.054,0.064,0.074,0.084,0.094\n";
        file << "1.0,0.0,1.0,0.17,0.14,0.37,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095\n";
        file << "0.0,1.0,1.0,0.1,0.2,0.33,0.016,0.026,0.036,0.046,0.056,0.066,0.076,0.086,0.096\n";
        file << "1.0,1.0,1.0,0.15,0.16,0.35,0.017,0.027,0.037,0.047,0.057,0.067,0.077,0.087,0.097\n";

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

    EXPECT_NO_THROW(loader.LoadFromCSV(test_file_path_, coordinates, field_data, grid_params));
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

    EXPECT_FLOAT_EQ(field_data[0].Bx, 0.1f);
    EXPECT_FLOAT_EQ(field_data[0].By, 0.2f);
    EXPECT_FLOAT_EQ(field_data[0].Bz, 0.3f);

    // 检查最后一个数据点
    EXPECT_FLOAT_EQ(coordinates[7].x, 1.0f);
    EXPECT_FLOAT_EQ(coordinates[7].y, 1.0f);
    EXPECT_FLOAT_EQ(coordinates[7].z, 1.0f);

    EXPECT_FLOAT_EQ(field_data[7].Bx, 0.15f);
    EXPECT_FLOAT_EQ(field_data[7].By, 0.16f);
    EXPECT_FLOAT_EQ(field_data[7].Bz, 0.35f);
}

// 测试文件不存在的情况
TEST_F(DataLoaderTest, FileNotFound) {
    DataLoader                     loader;
    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;
    GridParams                     grid_params;

    EXPECT_THROW(loader.LoadFromCSV("nonexistent_file.csv", coordinates, field_data, grid_params), std::runtime_error);
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

    EXPECT_THROW(loader.LoadFromCSV(invalid_file, coordinates, field_data, grid_params), std::runtime_error);

    std::remove(invalid_file.c_str());
}

// 测试空文件
TEST_F(DataLoaderTest, EmptyFile) {
    std::string   empty_file = "empty.csv";
    std::ofstream file(empty_file);
    file << "x,y,z,Bx,By,Bz,dBx_dx,dBx_dy,dBx_dz,dBy_dx,dBy_dy,dBy_dz,dBz_dx,dBz_dy,dBz_dz\n";  // 只有标题行
    file.close();

    DataLoader                     loader;
    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;
    GridParams                     grid_params;

    EXPECT_THROW(loader.LoadFromCSV(empty_file, coordinates, field_data, grid_params), std::runtime_error);

    std::remove(empty_file.c_str());
}

// 测试列索引设置
TEST_F(DataLoaderTest, CustomColumnIndices) {
    DataLoader loader;

    // 设置自定义列索引 (假设数据顺序不同)
    loader.SetColumnIndices({1, 2, 0}, {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                        14});  // x,y,z,Bx,By,Bz,dBx_dx,... -> y,z,x,Bx,By,Bz,dBx_dx,...

    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;
    GridParams                     grid_params;

    EXPECT_NO_THROW(loader.LoadFromCSV(test_file_path_, coordinates, field_data, grid_params));

    // 验证数据是否按自定义索引正确解析
    // 第一个数据行: 0.0,0.0,0.0,0.1,0.2,0.3,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09
    // 按索引 {1,2,0} 解析坐标: y=0.0, z=0.0, x=0.0 -> (0.0, 0.0, 0.0)
    // 按索引 {3,4,5,6,7,8,9,10,11,12,13,14} 解析磁场: Bx=0.1, By=0.2, Bz=0.3, dBx_dx=0.01, ...
    EXPECT_FLOAT_EQ(coordinates[0].x, 0.0f);
    EXPECT_FLOAT_EQ(coordinates[0].y, 0.0f);
    EXPECT_FLOAT_EQ(coordinates[0].z, 0.0f);

    EXPECT_FLOAT_EQ(field_data[0].Bx, 0.1f);
    EXPECT_FLOAT_EQ(field_data[0].By, 0.2f);
    EXPECT_FLOAT_EQ(field_data[0].Bz, 0.3f);
    EXPECT_FLOAT_EQ(field_data[0].dBx_dx, 0.01f);
    EXPECT_FLOAT_EQ(field_data[0].dBx_dy, 0.02f);
    EXPECT_FLOAT_EQ(field_data[0].dBx_dz, 0.03f);
    EXPECT_FLOAT_EQ(field_data[0].dBy_dx, 0.04f);
    EXPECT_FLOAT_EQ(field_data[0].dBy_dy, 0.05f);
    EXPECT_FLOAT_EQ(field_data[0].dBy_dz, 0.06f);
    EXPECT_FLOAT_EQ(field_data[0].dBz_dx, 0.07f);
    EXPECT_FLOAT_EQ(field_data[0].dBz_dy, 0.08f);
    EXPECT_FLOAT_EQ(field_data[0].dBz_dz, 0.09f);
}

}  // namespace test
}  // namespace p3d