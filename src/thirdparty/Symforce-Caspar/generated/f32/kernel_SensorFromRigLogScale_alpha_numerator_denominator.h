#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SensorFromRigLogScaleAlphaNumeratorDenominator(
    float* SensorFromRigLogScale_p_kp1,
    unsigned int SensorFromRigLogScale_p_kp1_num_alloc,
    float* SensorFromRigLogScale_r_k,
    unsigned int SensorFromRigLogScale_r_k_num_alloc,
    float* SensorFromRigLogScale_w,
    unsigned int SensorFromRigLogScale_w_num_alloc,
    float* const SensorFromRigLogScale_total_ag,
    float* const SensorFromRigLogScale_total_ac,
    size_t problem_size);

}  // namespace caspar