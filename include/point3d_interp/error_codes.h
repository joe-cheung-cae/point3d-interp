#ifndef POINTER3D_INTERP_ERROR_CODES_H
#define POINTER3D_INTERP_ERROR_CODES_H

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

inline std::string ErrorCodeToStringStd(ErrorCode code) {
    return std::string(ErrorCodeToString(code));
}

} // namespace p3d

#endif // POINTER3D_INTERP_ERROR_CODES_H