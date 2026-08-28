#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SensorFromRigLogScalePredDecreaseTimesTwo(
    float* SensorFromRigLogScale_step,
    unsigned int SensorFromRigLogScale_step_num_alloc,
    float* SensorFromRigLogScale_precond_diag,
    unsigned int SensorFromRigLogScale_precond_diag_num_alloc,
    const float* const diag,
    float* SensorFromRigLogScale_njtr,
    unsigned int SensorFromRigLogScale_njtr_num_alloc,
    float* const out_SensorFromRigLogScale_pred_dec,
    size_t problem_size);

}  // namespace caspar