#include "point3d_interp/error_codes.h"

namespace p3d {

const char* ErrorCodeToString(ErrorCode code) {
    switch (code) {
        case ErrorCode::Success:
            return "Success";
        case ErrorCode::FileNotFound:
            return "File not found";
        case ErrorCode::FileReadError:
            return "File read error";
        case ErrorCode::InvalidFileFormat:
            return "Invalid file format";
        case ErrorCode::InvalidGridData:
            return "Invalid grid data";
        case ErrorCode::MemoryAllocationError:
            return "Memory allocation error";
        case ErrorCode::CudaError:
            return "CUDA error";
        case ErrorCode::InvalidParameter:
            return "Invalid parameter";
        case ErrorCode::DataNotLoaded:
            return "Data not loaded";
        case ErrorCode::QueryOutOfBounds:
            return "Query point out of bounds";
        case ErrorCode::CudaNotAvailable:
            return "CUDA not available";
        case ErrorCode::CudaDeviceError:
            return "CUDA device error";
        case ErrorCode::UnknownError:
        default:
            return "Unknown error";
    }
}

} // namespace p3d