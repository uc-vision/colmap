#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void FixedRigPinholeResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    const double* const sensor_from_rig_log_scale,
    double* calib,
    unsigned int calib_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    const double* const reprojection_loss_scale,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double* out_sensor_from_rig_log_scale_jac,
    unsigned int out_sensor_from_rig_log_scale_jac_num_alloc,
    double* const out_sensor_from_rig_log_scale_njtr,
    double* const out_sensor_from_rig_log_scale_precond_diag,
    double* const out_sensor_from_rig_log_scale_precond_tril,
    double* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    double* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size);

}  // namespace caspar