#include "point3d_interp/interpolator_api.h"
#include <iostream>
#include <vector>

int main() {
    using namespace p3d;

    std::cout << "Testing structured grid extrapolation\n";

    // Create interpolator with nearest neighbor extrapolation
    MagneticFieldInterpolator interp(false, 0, InterpolationMethod::TricubicHermite,
                                     ExtrapolationMethod::NearestNeighbor);

    // Create simple 2x2x2 grid data
    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;

    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                Point3D coord(i * 1.0f, j * 1.0f, k * 1.0f);
                coordinates.push_back(coord);
                // Bx = x, By = y, Bz = z
                MagneticFieldData field(coord.x, coord.y, coord.z, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                                        1.0f);
                field_data.push_back(field);
            }
        }
    }

    ErrorCode err = interp.LoadFromMemory(coordinates.data(), field_data.data(), coordinates.size());
    if (err != ErrorCode::Success) {
        std::cerr << "Load failed\n";
        return 1;
    }

    // Test point inside domain
    Point3D             inside(0.5f, 0.5f, 0.5f);
    InterpolationResult result_inside;
    err = interp.Query(inside, result_inside);
    std::cout << "Inside point (0.5,0.5,0.5): B=(" << result_inside.data.Bx << "," << result_inside.data.By << ","
              << result_inside.data.Bz << ") valid=" << result_inside.valid << "\n";

    // Test point outside domain
    Point3D             outside(2.0f, 2.0f, 2.0f);  // Should clamp to (1,1,1)
    InterpolationResult result_outside;
    err = interp.Query(outside, result_outside);
    std::cout << "Outside point (2.0,2.0,2.0) with NearestNeighbor: B=(" << result_outside.data.Bx << ","
              << result_outside.data.By << "," << result_outside.data.Bz << ") valid=" << result_outside.valid << "\n";

    // Test with LinearExtrapolation
    MagneticFieldInterpolator interp_linear(false, 0, InterpolationMethod::TricubicHermite,
                                            ExtrapolationMethod::LinearExtrapolation);
    err = interp_linear.LoadFromMemory(coordinates.data(), field_data.data(), coordinates.size());
    InterpolationResult result_linear;
    err = interp_linear.Query(outside, result_linear);
    std::cout << "Outside point (2.0,2.0,2.0) with LinearExtrapolation: B=(" << result_linear.data.Bx << ","
              << result_linear.data.By << "," << result_linear.data.Bz << ") valid=" << result_linear.valid << "\n";

    // Test with None
    MagneticFieldInterpolator interp_none(false, 0, InterpolationMethod::TricubicHermite, ExtrapolationMethod::None);
    err = interp_none.LoadFromMemory(coordinates.data(), field_data.data(), coordinates.size());
    InterpolationResult result_none;
    err = interp_none.Query(outside, result_none);
    std::cout << "Outside point (2.0,2.0,2.0) with None: B=(" << result_none.data.Bx << "," << result_none.data.By
              << "," << result_none.data.Bz << ") valid=" << result_none.valid << "\n";

    return 0;
}