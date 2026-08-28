#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SensorFromRigLogScaleUpdateP(
    float* SensorFromRigLogScale_z,
    unsigned int SensorFromRigLogScale_z_num_alloc,
    float* SensorFromRigLogScale_p_k,
    unsigned int SensorFromRigLogScale_p_k_num_alloc,
    const float* const beta,
    float* out_SensorFromRigLogScale_p_kp1,
    unsigned int out_SensorFromRigLogScale_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar