#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigLogScalePriorResJac(
    const float* const sensor_from_rig_log_scale,
    const float* const target,
    const float* const sqrt_information,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_sensor_from_rig_log_scale_njtr,
    float* const out_sensor_from_rig_log_scale_precond_diag,
    float* const out_sensor_from_rig_log_scale_precond_tril,
    size_t problem_size);

}  // namespace caspar