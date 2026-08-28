#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SensorFromRigLogScaleRetract(
    float* SensorFromRigLogScale,
    unsigned int SensorFromRigLogScale_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_SensorFromRigLogScale_retracted,
    unsigned int out_SensorFromRigLogScale_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar