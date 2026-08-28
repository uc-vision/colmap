#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SensorFromRigLogScaleUpdateMp(
    float* SensorFromRigLogScale_r_k,
    unsigned int SensorFromRigLogScale_r_k_num_alloc,
    float* SensorFromRigLogScale_Mp,
    unsigned int SensorFromRigLogScale_Mp_num_alloc,
    const float* const beta,
    float* out_SensorFromRigLogScale_Mp_kp1,
    unsigned int out_SensorFromRigLogScale_Mp_kp1_num_alloc,
    float* out_SensorFromRigLogScale_w,
    unsigned int out_SensorFromRigLogScale_w_num_alloc,
    size_t problem_size);

}  // namespace caspar