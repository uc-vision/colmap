#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SensorFromRigLogScaleRetract(
    double* SensorFromRigLogScale,
    unsigned int SensorFromRigLogScale_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_SensorFromRigLogScale_retracted,
    unsigned int out_SensorFromRigLogScale_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar