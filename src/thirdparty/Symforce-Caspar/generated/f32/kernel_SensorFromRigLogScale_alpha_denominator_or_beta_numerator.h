#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SensorFromRigLogScaleAlphaDenominatorOrBetaNumerator(
    float* SensorFromRigLogScale_p_kp1,
    unsigned int SensorFromRigLogScale_p_kp1_num_alloc,
    float* SensorFromRigLogScale_w,
    unsigned int SensorFromRigLogScale_w_num_alloc,
    float* const SensorFromRigLogScale_out,
    size_t problem_size);

}  // namespace caspar