#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SensorFromRigLogScaleUpdateStep(
    float* SensorFromRigLogScale_step_k,
    unsigned int SensorFromRigLogScale_step_k_num_alloc,
    float* SensorFromRigLogScale_p_kp1,
    unsigned int SensorFromRigLogScale_p_kp1_num_alloc,
    const float* const alpha,
    float* out_SensorFromRigLogScale_step_kp1,
    unsigned int out_SensorFromRigLogScale_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar