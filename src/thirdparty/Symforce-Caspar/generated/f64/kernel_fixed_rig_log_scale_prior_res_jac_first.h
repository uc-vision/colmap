#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigLogScalePriorResJacFirst(
    const double* const sensor_from_rig_log_scale,
    const double* const target,
    const double* const sqrt_information,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* const out_sensor_from_rig_log_scale_njtr,
    double* const out_sensor_from_rig_log_scale_precond_diag,
    double* const out_sensor_from_rig_log_scale_precond_tril,
    size_t problem_size);

}  // namespace caspar