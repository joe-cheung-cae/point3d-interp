#include <gtest/gtest.h>
#include <point3d_interp/api.h>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

class ExporterTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create test data
        points = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}};

        field_data = {{1.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 0.0f, 0.1f},
                      {0.0f, 1.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f},
                      {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 0.1f, 0.1f, 0.0f, 0.0f},
                      {0.5f, 0.5f, 0.5f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f},
                      {2.0f, 2.0f, 2.0f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f}};

        // Create directory for test files in current directory
        temp_dir = "test_output";
        fs::create_directories(temp_dir);
        std::cout << "Test output directory: " << temp_dir << std::endl;
    }

    void TearDown() override {
        // Clean up test files
        // fs::remove_all(temp_dir);  // Commented out to keep files for inspection
    }

    std::vector<p3d::Point3D>           points;
    std::vector<p3d::MagneticFieldData> field_data;
    fs::path                            temp_dir;
};

TEST_F(ExporterTest, ExportInputPointsVTK) {
    p3d::MagneticFieldInterpolator interpolator;

    // Load test data
    auto err = interpolator.LoadFromMemory(points.data(), field_data.data(), points.size());
    ASSERT_EQ(err, p3d::ErrorCode::Success);

    // Export input points
    fs::path input_file = temp_dir / "input_points.vtk";
    std::cout << "Exporting input points to: " << input_file << std::endl;
    auto export_coordinates = interpolator.GetCoordinates();
    auto export_field_data = interpolator.GetFieldData();
    err = p3d::MagneticFieldInterpolator::ExportInputPoints(export_coordinates, export_field_data, p3d::ExportFormat::ParaviewVTK, input_file.string());
    ASSERT_EQ(err, p3d::ErrorCode::Success);

    // Check file exists
    ASSERT_TRUE(fs::exists(input_file));

    // Check file content (basic validation)
    std::ifstream file(input_file);
    ASSERT_TRUE(file.is_open());

    std::string line;
    std::getline(file, line);
    EXPECT_EQ(line, "# vtk DataFile Version 3.0");

    std::getline(file, line);
    EXPECT_EQ(line, "Input Sampling Points");

    std::getline(file, line);
    EXPECT_EQ(line, "ASCII");

    std::getline(file, line);
    EXPECT_EQ(line, "DATASET UNSTRUCTURED_GRID");

    std::getline(file, line);
    EXPECT_EQ(line, "POINTS 5 float");

    file.close();
}

TEST_F(ExporterTest, ExportOutputPointsVTK) {
    p3d::MagneticFieldInterpolator interpolator;

    // Load test data
    auto err = interpolator.LoadFromMemory(points.data(), field_data.data(), points.size());
    ASSERT_EQ(err, p3d::ErrorCode::Success);

    // Create query points
    std::vector<p3d::Point3D> query_points = {{0.5f, 0.5f, 0.5f}, {0.25f, 0.25f, 0.25f}};

    // Query interpolation results
    std::vector<p3d::InterpolationResult> results;
    err = interpolator.QueryBatch(query_points, results);
    ASSERT_EQ(err, p3d::ErrorCode::Success);
    ASSERT_EQ(results.size(), query_points.size());

    // Export output points
    fs::path output_file = temp_dir / "output_points.vtk";
    std::cout << "Exporting output points to: " << output_file << std::endl;
    err = interpolator.ExportOutputPoints(p3d::ExportFormat::ParaviewVTK, query_points, results, output_file.string());
    ASSERT_EQ(err, p3d::ErrorCode::Success);

    // Check file exists
    ASSERT_TRUE(fs::exists(output_file));

    // Check file content (basic validation)
    std::ifstream file(output_file);
    ASSERT_TRUE(file.is_open());

    std::string line;
    std::getline(file, line);
    EXPECT_EQ(line, "# vtk DataFile Version 3.0");

    std::getline(file, line);
    EXPECT_EQ(line, "Output Interpolation Points");

    std::getline(file, line);
    EXPECT_EQ(line, "ASCII");

    std::getline(file, line);
    EXPECT_EQ(line, "DATASET UNSTRUCTURED_GRID");

    std::getline(file, line);
    EXPECT_EQ(line, "POINTS 2 float");

    file.close();
}

TEST_F(ExporterTest, ExportWithoutDataLoaded) {
    // Since ExportInputPoints is now static and doesn't check for loaded data,
    // this test is no longer applicable. The static method just takes the data directly.
    SUCCEED();  // Skip this test
}

TEST_F(ExporterTest, ExportWithInvalidFormat) {
    // Try to export with invalid format (this will use Tecplot which is not implemented)
    // Note: Since Tecplot is not implemented, it should return InvalidParameter
    fs::path input_file = temp_dir / "invalid_format.vtk";
    auto err = p3d::MagneticFieldInterpolator::ExportInputPoints(points, field_data, static_cast<p3d::ExportFormat>(999), input_file.string());
    EXPECT_EQ(err, p3d::ErrorCode::InvalidParameter);
}