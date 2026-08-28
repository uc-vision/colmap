#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SensorFromRigLogScaleUpdateP(
    double* SensorFromRigLogScale_z,
    unsigned int SensorFromRigLogScale_z_num_alloc,
    double* SensorFromRigLogScale_p_k,
    unsigned int SensorFromRigLogScale_p_k_num_alloc,
    const double* const beta,
    double* out_SensorFromRigLogScale_p_kp1,
    unsigned int out_SensorFromRigLogScale_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar