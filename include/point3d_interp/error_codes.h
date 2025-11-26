#pragma once

#include <string>

namespace p3d {

enum class ErrorCode {
    Success = 0,
    FileNotFound,
    FileReadError,
    InvalidFileFormat,
    InvalidGridData,
    MemoryAllocationError,
    CudaError,
    InvalidParameter,
    DataNotLoaded,
    QueryOutOfBounds,
    CudaNotAvailable,
    CudaDeviceError,
    UnknownError
};

const char* ErrorCodeToString(ErrorCode code);

inline std::string ErrorCodeToStringStd(ErrorCode code) { return std::string(ErrorCodeToString(code)); }

}  // namespace p3d
