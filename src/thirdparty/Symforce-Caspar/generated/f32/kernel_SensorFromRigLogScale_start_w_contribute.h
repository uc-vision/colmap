#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SensorFromRigLogScaleStartWContribute(
    float* SensorFromRigLogScale_precond_diag,
    unsigned int SensorFromRigLogScale_precond_diag_num_alloc,
    const float* const diag,
    float* SensorFromRigLogScale_p,
    unsigned int SensorFromRigLogScale_p_num_alloc,
    float* out_SensorFromRigLogScale_w,
    unsigned int out_SensorFromRigLogScale_w_num_alloc,
    size_t problem_size);

}  // namespace caspar