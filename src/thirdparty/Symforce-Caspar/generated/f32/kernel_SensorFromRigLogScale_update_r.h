#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SensorFromRigLogScaleUpdateR(
    float* SensorFromRigLogScale_r_k,
    unsigned int SensorFromRigLogScale_r_k_num_alloc,
    float* SensorFromRigLogScale_w,
    unsigned int SensorFromRigLogScale_w_num_alloc,
    const float* const negalpha,
    float* out_SensorFromRigLogScale_r_kp1,
    unsigned int out_SensorFromRigLogScale_r_kp1_num_alloc,
    float* const out_SensorFromRigLogScale_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar