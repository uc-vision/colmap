#include "kernel_row_fixed_rig_pinhole_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) RowFixedRigPinholeResJacKernel(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_calibration,
    unsigned int sensor_calibration_num_alloc,
    SharedIndex* sensor_calibration_indices,
    const double* const sensor_from_rig_log_scale,
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
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex sensor_calibration_indices_loc[1024];
  sensor_calibration_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? sensor_calibration_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_sensor_from_rig_log_scale_njtr_local[1];

  __shared__ double out_sensor_from_rig_log_scale_precond_diag_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86;
  LoadShared<2, double, double>(sensor_calibration,
                                8 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r0,
                        r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r1);
    r1 = 1.00000000000000008e-15;
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r5, r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = 2.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  LoadShared<2, double, double>(sensor_calibration,
                                2 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r10,
                        r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r8 * r11;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r13, r14);
  };
  __syncthreads();
  LoadShared<2, double, double>(sensor_calibration,
                                0 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r15,
                        r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r17 = r14 * r15;
    r18 = r12 + r17;
    r19 = r13 * r16;
    r18 = fma(r9, r10, r18);
    r18 = fma(r4, r19, r18);
    r20 = r7 * r18;
    r21 = fma(r9, r15, r13 * r11);
    r22 = r14 * r10;
    r21 = fma(r4, r22, r21);
    r21 = fma(r8, r16, r21);
    r20 = r20 * r21;
    r22 = fma(r14, r16, r13 * r15);
    r22 = fma(r8, r10, r22);
    r22 = fma(r9, r11, r4 * r22);
    r23 = r8 * r15;
    r23 = fma(r4, r23, r14 * r11);
    r23 = fma(r9, r16, r23);
    r23 = fma(r13, r10, r23);
    r24 = -2.00000000000000000e+00;
    r25 = r23 * r24;
    r26 = fma(r22, r25, r20);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r28 = r15 * r15;
    r28 = r28 * r24;
    r29 = 1.00000000000000000e+00;
    r30 = r16 * r16;
    r30 = fma(r24, r30, r29);
    r31 = r28 + r30;
    r32 = fma(r27, r31, r5 * r26);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r33, r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r35 = r15 * r10;
    r35 = r35 * r7;
    r36 = r16 * r11;
    r36 = fma(r24, r36, r35);
    r37 = r15 * r11;
    r38 = r16 * r10;
    r38 = r38 * r7;
    r37 = fma(r7, r37, r38);
  };
  LoadShared<2, double, double>(sensor_calibration,
                                6 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r39,
                        r40);
  };
  __syncthreads();
  LoadUnique<1, double, double>(
      sensor_from_rig_log_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r41);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r42 = 2.71828182845904523536;
    r41 = pow(r42, r41);
    r39 = r39 * r41;
    r42 = r7 * r18;
    r42 = r42 * r23;
    r43 = r7 * r22;
    r44 = fma(r21, r43, r42);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r45);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r46 = r23 * r25;
    r47 = r29 + r46;
    r48 = r21 * r21;
    r48 = r48 * r24;
    r47 = r47 + r48;
    r32 = fma(r33, r36, r32);
    r32 = fma(r34, r37, r32);
    r32 = r32 + r39;
    r32 = fma(r6, r44, r32);
    r32 = fma(r45, r47, r32);
    r49 = copysign(1.0, r32);
    r49 = fma(r1, r49, r32);
    r1 = 1.0 / r49;
    r20 = fma(r23, r43, r20);
    r32 = r10 * r10;
    r32 = r24 * r32;
    r30 = r32 + r30;
    r50 = fma(r33, r30, r45 * r20);
    r51 = r16 * r11;
    r51 = fma(r7, r51, r35);
    r35 = r10 * r11;
    r52 = r15 * r16;
    r52 = r52 * r7;
    r35 = fma(r24, r35, r52);
  };
  LoadShared<2, double, double>(sensor_calibration,
                                4 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r53,
                        r54);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r46 = r29 + r46;
    r55 = r18 * r18;
    r55 = r55 * r24;
    r46 = r46 + r55;
    r56 = r7 * r21;
    r56 = r56 * r23;
    r57 = r18 * r22;
    r57 = fma(r24, r57, r56);
    r50 = fma(r27, r51, r50);
    r50 = fma(r34, r35, r50);
    r50 = fma(r53, r41, r50);
    r50 = fma(r5, r46, r50);
    r50 = fma(r6, r57, r50);
    r50 = r40 * r50;
    r2 = fma(r1, r50, r2);
  };
  LoadUnique<1, double, double>(
      reprojection_loss_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r58);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r58 = r58 * r58;
    r58 = 1.0 / r58;
  };
  LoadShared<1, double, double>(sensor_calibration,
                                10 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r59);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = fma(r3, r4, r59);
    r55 = r29 + r55;
    r55 = r55 + r48;
    r32 = r29 + r32;
    r32 = r32 + r28;
    r34 = fma(r34, r32, r6 * r55);
    r28 = r10 * r11;
    r28 = fma(r7, r28, r52);
    r52 = r15 * r11;
    r52 = fma(r24, r52, r38);
    r38 = r21 * r22;
    r38 = fma(r24, r38, r42);
    r56 = fma(r18, r43, r56);
    r34 = fma(r33, r28, r34);
    r34 = fma(r27, r52, r34);
    r34 = fma(r54, r41, r34);
    r34 = fma(r45, r38, r34);
    r34 = fma(r5, r56, r34);
    r34 = r0 * r34;
    r3 = fma(r1, r34, r3);
    r27 = fma(r3, r3, r2 * r2);
    r27 = fma(r27, r58, r29);
    r33 = sqrt(r27);
    r33 = r29 + r33;
    r29 = 1.0 / r33;
    r29 = r7 * r29;
    r42 = sqrt(r29);
    r48 = r2 * r42;
    r59 = r3 * r42;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r48, r59);
    r59 = r7 * r3;
    r60 = r22 * r24;
    r61 = 5.00000000000000000e-01;
    r62 = fma(r61, r17, r61 * r12);
    r63 = -5.00000000000000000e-01;
    r64 = r9 * r61;
    r62 = fma(r63, r19, r62);
    r62 = fma(r10, r64, r62);
    r65 = r13 * r11;
    r66 = r9 * r15;
    r66 = fma(r63, r66, r63 * r65);
    r65 = r8 * r16;
    r66 = fma(r63, r65, r66);
    r67 = r14 * r10;
    r66 = fma(r61, r67, r66);
    r60 = fma(r66, r25, r62 * r60);
    r67 = r7 * r18;
    r65 = r13 * r15;
    r68 = r14 * r16;
    r68 = fma(r63, r68, r63 * r65);
    r65 = r8 * r10;
    r68 = fma(r63, r65, r68);
    r68 = fma(r11, r64, r68);
    r67 = r67 * r68;
    r65 = r7 * r21;
    r69 = r14 * r11;
    r70 = r8 * r15;
    r70 = fma(r61, r70, r63 * r69);
    r69 = r9 * r16;
    r70 = fma(r63, r69, r70);
    r71 = r13 * r10;
    r70 = fma(r63, r71, r70);
    r65 = fma(r70, r65, r67);
    r60 = r60 + r65;
    r71 = r7 * r23;
    r71 = r71 * r70;
    r69 = r7 * r18;
    r69 = fma(r62, r69, r71);
    r72 = r7 * r21;
    r72 = r72 * r66;
    r73 = r68 * r43;
    r74 = r72 + r73;
    r75 = r69 + r74;
    r75 = fma(r6, r75, r5 * r60);
    r60 = r23 * r62;
    r76 = -4.00000000000000000e+00;
    r60 = r60 * r76;
    r77 = r21 * r76;
    r78 = r68 * r77;
    r79 = r60 + r78;
    r75 = fma(r45, r79, r75);
    r49 = r49 * r49;
    r49 = 1.0 / r49;
    r49 = r4 * r49;
    r79 = r75 * r49;
    r80 = r18 * r70;
    r81 = r76 * r80;
    r78 = r78 + r81;
    r82 = r7 * r21;
    r82 = r82 * r62;
    r83 = fma(r70, r43, r82);
    r84 = r7 * r23;
    r84 = r84 * r68;
    r85 = r7 * r18;
    r85 = fma(r66, r85, r84);
    r83 = r83 + r85;
    r83 = fma(r5, r83, r6 * r78);
    r78 = r21 * r24;
    r86 = r22 * r24;
    r86 = r86 * r68;
    r78 = fma(r66, r78, r86);
    r78 = r78 + r69;
    r83 = fma(r45, r78, r83);
    r78 = r0 * r83;
    r78 = fma(r1, r78, r34 * r79);
    r79 = r7 * r2;
    r81 = r60 + r81;
    r82 = r84 + r82;
    r84 = r18 * r24;
    r82 = fma(r66, r84, r82);
    r60 = r22 * r24;
    r82 = fma(r70, r60, r82);
    r82 = fma(r6, r82, r5 * r81);
    r81 = r7 * r23;
    r62 = fma(r62, r43, r66 * r81);
    r62 = r62 + r65;
    r82 = fma(r45, r62, r82);
    r62 = r40 * r82;
    r50 = r49 * r50;
    r62 = fma(r75, r50, r1 * r62);
    r79 = fma(r62, r79, r78 * r59);
    r59 = r2 * r79;
    r58 = r63 * r58;
    r29 = rsqrt(r29);
    r33 = r33 * r33;
    r33 = 1.0 / r33;
    r27 = rsqrt(r27);
    r58 = r58 * r29;
    r58 = r58 * r33;
    r58 = r58 * r27;
    r62 = fma(r62, r42, r58 * r59);
    r59 = r3 * r58;
    r78 = fma(r79, r59, r78 * r42);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r62, r78);
    r27 = r7 * r2;
    r33 = r7 * r23;
    r29 = r13 * r11;
    r81 = r8 * r16;
    r81 = fma(r61, r81, r61 * r29);
    r29 = r14 * r10;
    r81 = fma(r63, r29, r81);
    r81 = fma(r15, r64, r81);
    r33 = r33 * r81;
    r29 = r9 * r10;
    r12 = fma(r63, r12, r63 * r29);
    r12 = fma(r63, r17, r12);
    r12 = fma(r61, r19, r12);
    r19 = fma(r12, r43, r33);
    r19 = r19 + r65;
    r65 = fma(r70, r25, r86);
    r17 = r7 * r18;
    r17 = r17 * r12;
    r29 = r7 * r21;
    r29 = fma(r81, r29, r17);
    r65 = r65 + r29;
    r65 = fma(r5, r65, r6 * r19);
    r19 = r23 * r68;
    r19 = r19 * r76;
    r60 = r12 * r77;
    r84 = r19 + r60;
    r65 = fma(r45, r84, r65);
    r84 = r18 * r76;
    r84 = r84 * r81;
    r19 = r19 + r84;
    r73 = r71 + r73;
    r73 = r73 + r29;
    r73 = fma(r45, r73, r5 * r19);
    r19 = r22 * r24;
    r19 = fma(r24, r80, r81 * r19);
    r29 = r7 * r23;
    r71 = r7 * r21;
    r71 = r71 * r68;
    r29 = fma(r12, r29, r71);
    r19 = r19 + r29;
    r73 = fma(r6, r19, r73);
    r19 = r40 * r73;
    r19 = fma(r1, r19, r65 * r50);
    r69 = r7 * r3;
    r60 = r84 + r60;
    r33 = r67 + r33;
    r67 = r21 * r24;
    r33 = fma(r70, r67, r33);
    r70 = r22 * r24;
    r33 = fma(r12, r70, r33);
    r33 = fma(r45, r33, r6 * r60);
    r81 = fma(r81, r43, r7 * r80);
    r81 = r81 + r29;
    r33 = fma(r5, r81, r33);
    r81 = r0 * r33;
    r80 = r65 * r49;
    r80 = fma(r34, r80, r1 * r81);
    r69 = fma(r80, r69, r19 * r27);
    r27 = r2 * r69;
    r19 = fma(r19, r42, r58 * r27);
    r80 = fma(r69, r59, r80 * r42);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r19, r80);
    r27 = r7 * r3;
    r81 = r7 * r21;
    r60 = r14 * r11;
    r70 = r8 * r15;
    r70 = fma(r63, r70, r61 * r60);
    r60 = r13 * r10;
    r70 = fma(r61, r60, r70);
    r70 = fma(r16, r64, r70);
    r81 = fma(r70, r43, r12 * r81);
    r81 = r81 + r85;
    r64 = r7 * r18;
    r64 = r64 * r70;
    r71 = r71 + r64;
    r60 = r22 * r24;
    r71 = fma(r66, r60, r71);
    r71 = fma(r12, r25, r71);
    r71 = fma(r5, r71, r6 * r81);
    r81 = r23 * r66;
    r81 = r81 * r76;
    r77 = r70 * r77;
    r25 = r81 + r77;
    r71 = fma(r45, r25, r71);
    r25 = r71 * r49;
    r68 = r18 * r68;
    r68 = r68 * r76;
    r77 = r68 + r77;
    r76 = r21 * r24;
    r60 = r22 * r24;
    r60 = fma(r70, r60, r12 * r76);
    r60 = r60 + r85;
    r60 = fma(r45, r60, r6 * r77);
    r77 = r7 * r23;
    r77 = r77 * r70;
    r17 = r17 + r77;
    r17 = r17 + r74;
    r60 = fma(r5, r17, r60);
    r17 = r0 * r60;
    r17 = fma(r1, r17, r34 * r25);
    r25 = r7 * r2;
    r68 = r81 + r68;
    r86 = r72 + r86;
    r72 = r18 * r24;
    r86 = fma(r12, r72, r86);
    r86 = r86 + r77;
    r86 = fma(r6, r86, r5 * r68);
    r43 = fma(r66, r43, r64);
    r43 = r43 + r29;
    r86 = fma(r45, r43, r86);
    r43 = r40 * r86;
    r43 = fma(r71, r50, r1 * r43);
    r25 = fma(r43, r25, r17 * r27);
    r27 = r2 * r25;
    r43 = fma(r43, r42, r58 * r27);
    r17 = fma(r25, r59, r17 * r42);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r43, r17);
    r27 = r7 * r3;
    r45 = r0 * r28;
    r29 = r36 * r49;
    r29 = fma(r34, r29, r1 * r45);
    r45 = r7 * r2;
    r66 = r40 * r30;
    r66 = fma(r36, r50, r1 * r66);
    r45 = fma(r66, r45, r29 * r27);
    r27 = r2 * r45;
    r66 = fma(r66, r42, r58 * r27);
    r29 = fma(r29, r42, r45 * r59);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r66, r29);
    r27 = r7 * r2;
    r64 = r40 * r35;
    r64 = fma(r37, r50, r1 * r64);
    r6 = r7 * r3;
    r68 = r37 * r49;
    r5 = r0 * r32;
    r5 = fma(r1, r5, r34 * r68);
    r6 = fma(r5, r6, r64 * r27);
    r27 = r2 * r6;
    r64 = fma(r64, r42, r58 * r27);
    r5 = fma(r5, r42, r6 * r59);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r64, r5);
    r27 = r7 * r3;
    r68 = r0 * r52;
    r72 = r31 * r49;
    r72 = fma(r34, r72, r1 * r68);
    r68 = r7 * r2;
    r77 = r40 * r51;
    r77 = fma(r1, r77, r31 * r50);
    r68 = fma(r77, r68, r72 * r27);
    r27 = r2 * r68;
    r77 = fma(r77, r42, r58 * r27);
    r72 = fma(r68, r59, r72 * r42);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r77, r72);
    r27 = r4 * r3;
    r27 = r27 * r78;
    r48 = r4 * r48;
    r27 = fma(r62, r48, r42 * r27);
    r12 = r4 * r3;
    r12 = r12 * r80;
    r12 = fma(r19, r48, r42 * r12);
    WriteSum2<double, double>((double*)inout_shared, r27, r12);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = r4 * r3;
    r12 = r12 * r17;
    r12 = fma(r42, r12, r43 * r48);
    r27 = r4 * r3;
    r27 = r27 * r29;
    r27 = fma(r66, r48, r42 * r27);
    WriteSum2<double, double>((double*)inout_shared, r12, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r4 * r3;
    r27 = r27 * r5;
    r27 = fma(r42, r27, r64 * r48);
    r12 = r4 * r3;
    r12 = r12 * r72;
    r12 = fma(r42, r12, r77 * r48);
    WriteSum2<double, double>((double*)inout_shared, r27, r12);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r78, r78, r62 * r62);
    r27 = fma(r19, r19, r80 * r80);
    WriteSum2<double, double>((double*)inout_shared, r12, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r43, r43, r17 * r17);
    r12 = fma(r29, r29, r66 * r66);
    WriteSum2<double, double>((double*)inout_shared, r27, r12);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r5, r5, r64 * r64);
    r27 = fma(r77, r77, r72 * r72);
    WriteSum2<double, double>((double*)inout_shared, r12, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r62, r19, r78 * r80);
    r12 = fma(r78, r17, r62 * r43);
    WriteSum2<double, double>((double*)inout_shared, r27, r12);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r62, r66, r78 * r29);
    r27 = fma(r62, r64, r78 * r5);
    WriteSum2<double, double>((double*)inout_shared, r12, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fma(r62, r77, r78 * r72);
    r78 = fma(r19, r43, r80 * r17);
    WriteSum2<double, double>((double*)inout_shared, r62, r78);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = fma(r19, r66, r80 * r29);
    r62 = fma(r19, r64, r80 * r5);
    WriteSum2<double, double>((double*)inout_shared, r78, r62);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r19, r77, r80 * r72);
    r80 = fma(r43, r66, r17 * r29);
    WriteSum2<double, double>((double*)inout_shared, r19, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r17, r5, r43 * r64);
    r17 = fma(r17, r72, r43 * r77);
    WriteSum2<double, double>((double*)inout_shared, r80, r17);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = fma(r66, r64, r29 * r5);
    r66 = fma(r66, r77, r29 * r72);
    WriteSum2<double, double>((double*)inout_shared, r17, r66);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fma(r64, r77, r5 * r72);
    WriteSum1<double, double>((double*)inout_shared, r77);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r53 * r40;
    r77 = r77 * r41;
    r77 = fma(r1, r77, r50 * r39);
    r64 = r7 * r3;
    r72 = r54 * r0;
    r72 = r72 * r41;
    r41 = r49 * r34;
    r41 = fma(r39, r41, r1 * r72);
    r72 = r7 * r2;
    r72 = fma(r77, r72, r41 * r64);
    r64 = r2 * r72;
    r64 = fma(r58, r64, r77 * r42);
    r41 = fma(r41, r42, r72 * r59);
    WriteIdx2<1024, double, double, double2>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r64,
        r41);
    r77 = r4 * r3;
    r77 = r77 * r41;
    r77 = fma(r64, r48, r42 * r77);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_njtr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r77);
  if (global_thread_idx < problem_size) {
    r64 = fma(r64, r64, r41 * r41);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_precond_diag_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r64);
  if (global_thread_idx < problem_size) {
    r64 = r40 * r46;
    r64 = fma(r26, r50, r1 * r64);
    r41 = r7 * r2;
    r77 = r7 * r3;
    r39 = r26 * r49;
    r5 = r0 * r56;
    r5 = fma(r1, r5, r34 * r39);
    r77 = fma(r5, r77, r64 * r41);
    r41 = r2 * r77;
    r41 = fma(r58, r41, r64 * r42);
    r5 = fma(r77, r59, r5 * r42);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r41, r5);
    r64 = r40 * r57;
    r64 = fma(r44, r50, r1 * r64);
    r39 = r7 * r2;
    r66 = r7 * r3;
    r17 = r0 * r55;
    r29 = r44 * r49;
    r29 = fma(r34, r29, r1 * r17);
    r66 = fma(r29, r66, r64 * r39);
    r39 = r2 * r66;
    r39 = fma(r58, r39, r64 * r42);
    r29 = fma(r29, r42, r66 * r59);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r39,
                                             r29);
    r64 = r40 * r20;
    r50 = fma(r47, r50, r1 * r64);
    r64 = r7 * r2;
    r17 = r7 * r3;
    r80 = r47 * r49;
    r43 = r0 * r38;
    r43 = fma(r1, r43, r34 * r80);
    r17 = fma(r43, r17, r50 * r64);
    r64 = r2 * r17;
    r64 = fma(r58, r64, r50 * r42);
    r43 = fma(r43, r42, r17 * r59);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r64,
                                             r43);
    r59 = r4 * r3;
    r59 = r59 * r5;
    r59 = fma(r41, r48, r42 * r59);
    r50 = r4 * r3;
    r50 = r50 * r29;
    r50 = fma(r42, r50, r39 * r48);
    WriteSum2<double, double>((double*)inout_shared, r59, r50);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r4 * r3;
    r50 = r50 * r43;
    r50 = fma(r42, r50, r64 * r48);
    WriteSum1<double, double>((double*)inout_shared, r50);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fma(r41, r41, r5 * r5);
    r48 = fma(r39, r39, r29 * r29);
    WriteSum2<double, double>((double*)inout_shared, r50, r48);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fma(r43, r43, r64 * r64);
    WriteSum1<double, double>((double*)inout_shared, r48);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fma(r41, r39, r5 * r29);
    r41 = fma(r41, r64, r5 * r43);
    WriteSum2<double, double>((double*)inout_shared, r48, r41);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fma(r29, r43, r39 * r64);
    WriteSum1<double, double>((double*)inout_shared, r43);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_sensor_from_rig_log_scale_njtr_local,
                        out_sensor_from_rig_log_scale_njtr,
                        1);
  SumFlushFinal<double>(out_sensor_from_rig_log_scale_precond_diag_local,
                        out_sensor_from_rig_log_scale_precond_diag,
                        1);
}

void RowFixedRigPinholeResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_calibration,
    unsigned int sensor_calibration_num_alloc,
    SharedIndex* sensor_calibration_indices,
    const double* const sensor_from_rig_log_scale,
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
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  RowFixedRigPinholeResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_calibration,
      sensor_calibration_num_alloc,
      sensor_calibration_indices,
      sensor_from_rig_log_scale,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      reprojection_loss_scale,
      out_res,
      out_res_num_alloc,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_sensor_from_rig_log_scale_jac,
      out_sensor_from_rig_log_scale_jac_num_alloc,
      out_sensor_from_rig_log_scale_njtr,
      out_sensor_from_rig_log_scale_precond_diag,
      out_sensor_from_rig_log_scale_precond_tril,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar