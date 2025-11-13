#!/bin/bash

# Script to format all C/C++/CUDA source and header files in the project using clang-format

# Find all relevant files
find . -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) \
    ! -path "./build/*" \
    ! -path "./_deps/*" \
    -exec clang-format -i {} +

echo "Formatting complete."