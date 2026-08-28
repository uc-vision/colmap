#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SensorFromRigLogScaleAlphaNumeratorDenominator(
    double* SensorFromRigLogScale_p_kp1,
    unsigned int SensorFromRigLogScale_p_kp1_num_alloc,
    double* SensorFromRigLogScale_r_k,
    unsigned int SensorFromRigLogScale_r_k_num_alloc,
    double* SensorFromRigLogScale_w,
    unsigned int SensorFromRigLogScale_w_num_alloc,
    double* const SensorFromRigLogScale_total_ag,
    double* const SensorFromRigLogScale_total_ac,
    size_t problem_size);

}  // namespace caspar