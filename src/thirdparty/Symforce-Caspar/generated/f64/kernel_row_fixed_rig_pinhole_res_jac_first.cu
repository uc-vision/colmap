#include "kernel_row_fixed_rig_pinhole_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) RowFixedRigPinholeResJacFirstKernel(
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
    double* const out_rTr,
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

  __shared__ double out_rTr_local[1];

  __shared__ double out_sensor_from_rig_log_scale_njtr_local[1];

  __shared__ double out_sensor_from_rig_log_scale_precond_diag_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85;
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
    r22 = r8 * r15;
    r22 = fma(r4, r22, r14 * r11);
    r22 = fma(r9, r16, r22);
    r22 = fma(r13, r10, r22);
    r23 = fma(r14, r16, r13 * r15);
    r23 = fma(r8, r10, r23);
    r23 = fma(r9, r11, r4 * r23);
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
    r42 = r42 * r22;
    r43 = r7 * r21;
    r43 = fma(r23, r43, r42);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r44);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r45 = r22 * r22;
    r45 = r45 * r24;
    r46 = r29 + r45;
    r47 = r21 * r21;
    r47 = r47 * r24;
    r46 = r46 + r47;
    r32 = fma(r33, r36, r32);
    r32 = fma(r34, r37, r32);
    r32 = r32 + r39;
    r32 = fma(r6, r43, r32);
    r32 = fma(r44, r46, r32);
    r48 = copysign(1.0, r32);
    r48 = fma(r1, r48, r32);
    r1 = 1.0 / r48;
    r32 = r7 * r22;
    r32 = fma(r23, r32, r20);
    r20 = r10 * r10;
    r20 = r24 * r20;
    r30 = r20 + r30;
    r49 = fma(r33, r30, r44 * r32);
    r50 = r16 * r11;
    r50 = fma(r7, r50, r35);
    r35 = r10 * r11;
    r51 = r15 * r16;
    r51 = r51 * r7;
    r35 = fma(r24, r35, r51);
  };
  LoadShared<2, double, double>(sensor_calibration,
                                4 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r52,
                        r53);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r45 = r29 + r45;
    r54 = r18 * r18;
    r54 = r54 * r24;
    r45 = r45 + r54;
    r55 = r7 * r21;
    r55 = r55 * r22;
    r56 = fma(r18, r25, r55);
    r49 = fma(r27, r50, r49);
    r49 = fma(r34, r35, r49);
    r49 = fma(r52, r41, r49);
    r49 = fma(r5, r45, r49);
    r49 = fma(r6, r56, r49);
    r49 = r40 * r49;
    r2 = fma(r1, r49, r2);
  };
  LoadUnique<1, double, double>(
      reprojection_loss_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r57);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r57 = r57 * r57;
    r57 = 1.0 / r57;
  };
  LoadShared<1, double, double>(sensor_calibration,
                                10 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r58);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = fma(r3, r4, r58);
    r54 = r29 + r54;
    r54 = r54 + r47;
    r20 = r29 + r20;
    r20 = r20 + r28;
    r34 = fma(r34, r20, r6 * r54);
    r28 = r10 * r11;
    r28 = fma(r7, r28, r51);
    r51 = r15 * r11;
    r51 = fma(r24, r51, r38);
    r42 = fma(r21, r25, r42);
    r38 = r7 * r18;
    r38 = fma(r23, r38, r55);
    r34 = fma(r33, r28, r34);
    r34 = fma(r27, r51, r34);
    r34 = fma(r53, r41, r34);
    r34 = fma(r44, r42, r34);
    r34 = fma(r5, r38, r34);
    r34 = r0 * r34;
    r3 = fma(r1, r34, r3);
    r27 = fma(r3, r3, r2 * r2);
    r27 = fma(r27, r57, r29);
    r33 = sqrt(r27);
    r33 = r29 + r33;
    r29 = 1.0 / r33;
    r55 = r7 * r29;
    r47 = sqrt(r55);
    r58 = r2 * r47;
    r59 = r3 * r47;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r58, r59);
    r59 = r7 * r2;
    r59 = r59 * r2;
    r60 = r3 * r29;
    r61 = r7 * r3;
    r60 = fma(r61, r60, r29 * r59);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r60);
  if (global_thread_idx < problem_size) {
    r60 = r7 * r2;
    r59 = 5.00000000000000000e-01;
    r62 = fma(r59, r17, r59 * r12);
    r63 = -5.00000000000000000e-01;
    r64 = r9 * r59;
    r62 = fma(r63, r19, r62);
    r62 = fma(r10, r64, r62);
    r65 = r22 * r62;
    r66 = -4.00000000000000000e+00;
    r65 = r65 * r66;
    r67 = r14 * r11;
    r68 = r8 * r15;
    r68 = fma(r59, r68, r63 * r67);
    r67 = r9 * r16;
    r68 = fma(r63, r67, r68);
    r69 = r13 * r10;
    r68 = fma(r63, r69, r68);
    r69 = r18 * r68;
    r67 = r66 * r69;
    r70 = r65 + r67;
    r71 = r7 * r22;
    r72 = r13 * r15;
    r73 = r14 * r16;
    r73 = fma(r63, r73, r63 * r72);
    r72 = r8 * r10;
    r73 = fma(r63, r72, r73);
    r73 = fma(r11, r64, r73);
    r71 = r71 * r73;
    r72 = r7 * r21;
    r72 = r72 * r62;
    r74 = r71 + r72;
    r75 = r18 * r24;
    r76 = r13 * r11;
    r77 = r9 * r15;
    r77 = fma(r63, r77, r63 * r76);
    r76 = r8 * r16;
    r77 = fma(r63, r76, r77);
    r78 = r14 * r10;
    r77 = fma(r59, r78, r77);
    r74 = fma(r77, r75, r74);
    r74 = fma(r68, r25, r74);
    r74 = fma(r6, r74, r5 * r70);
    r70 = r7 * r23;
    r75 = r22 * r77;
    r70 = fma(r7, r75, r62 * r70);
    r78 = r7 * r18;
    r78 = r78 * r73;
    r76 = r7 * r21;
    r76 = fma(r68, r76, r78);
    r70 = r70 + r76;
    r74 = fma(r44, r70, r74);
    r70 = r40 * r74;
    r79 = fma(r62, r25, r24 * r75);
    r79 = r79 + r76;
    r80 = r7 * r22;
    r80 = r80 * r68;
    r81 = r7 * r18;
    r81 = fma(r62, r81, r80);
    r62 = r7 * r21;
    r62 = r62 * r77;
    r82 = r7 * r23;
    r82 = r82 * r73;
    r83 = r62 + r82;
    r84 = r81 + r83;
    r84 = fma(r6, r84, r5 * r79);
    r79 = r21 * r66;
    r85 = r73 * r79;
    r65 = r65 + r85;
    r84 = fma(r44, r65, r84);
    r48 = r48 * r48;
    r48 = 1.0 / r48;
    r48 = r4 * r48;
    r49 = r48 * r49;
    r70 = fma(r84, r49, r1 * r70);
    r65 = r84 * r48;
    r67 = r85 + r67;
    r85 = r7 * r23;
    r85 = fma(r68, r85, r72);
    r72 = r7 * r18;
    r72 = fma(r77, r72, r71);
    r85 = r85 + r72;
    r85 = fma(r5, r85, r6 * r67);
    r67 = r21 * r24;
    r71 = r73 * r25;
    r67 = fma(r77, r67, r71);
    r67 = r67 + r81;
    r85 = fma(r44, r67, r85);
    r67 = r0 * r85;
    r67 = fma(r1, r67, r34 * r65);
    r60 = fma(r67, r61, r70 * r60);
    r57 = r63 * r57;
    r55 = rsqrt(r55);
    r33 = r33 * r33;
    r33 = 1.0 / r33;
    r27 = rsqrt(r27);
    r57 = r57 * r55;
    r57 = r57 * r33;
    r57 = r57 * r27;
    r60 = r60 * r57;
    r70 = fma(r70, r47, r2 * r60);
    r60 = fma(r3, r60, r67 * r47);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r70, r60);
    r67 = r7 * r2;
    r27 = r7 * r22;
    r33 = r13 * r11;
    r55 = r8 * r16;
    r55 = fma(r59, r55, r59 * r33);
    r33 = r14 * r10;
    r55 = fma(r63, r33, r55);
    r55 = fma(r15, r64, r55);
    r27 = r27 * r55;
    r33 = r7 * r23;
    r65 = r9 * r10;
    r12 = fma(r63, r12, r63 * r65);
    r12 = fma(r63, r17, r12);
    r12 = fma(r59, r19, r12);
    r33 = fma(r12, r33, r27);
    r33 = r33 + r76;
    r76 = r22 * r24;
    r76 = fma(r68, r76, r71);
    r19 = r7 * r18;
    r19 = r19 * r12;
    r17 = r7 * r21;
    r17 = fma(r55, r17, r19);
    r76 = r76 + r17;
    r76 = fma(r5, r76, r6 * r33);
    r33 = r22 * r73;
    r33 = r33 * r66;
    r65 = r12 * r79;
    r81 = r33 + r65;
    r76 = fma(r44, r81, r76);
    r81 = r18 * r66;
    r81 = r81 * r55;
    r33 = r33 + r81;
    r82 = r80 + r82;
    r82 = r82 + r17;
    r82 = fma(r44, r82, r5 * r33);
    r33 = fma(r55, r25, r24 * r69);
    r17 = r7 * r22;
    r80 = r7 * r21;
    r80 = r80 * r73;
    r17 = fma(r12, r17, r80);
    r33 = r33 + r17;
    r82 = fma(r6, r33, r82);
    r33 = r40 * r82;
    r33 = fma(r1, r33, r76 * r49);
    r65 = r81 + r65;
    r27 = r78 + r27;
    r78 = r21 * r24;
    r27 = fma(r68, r78, r27);
    r27 = fma(r12, r25, r27);
    r27 = fma(r44, r27, r6 * r65);
    r65 = r7 * r23;
    r69 = fma(r7, r69, r55 * r65);
    r69 = r69 + r17;
    r27 = fma(r5, r69, r27);
    r69 = r0 * r27;
    r65 = r76 * r48;
    r65 = fma(r34, r65, r1 * r69);
    r67 = fma(r65, r61, r33 * r67);
    r69 = r2 * r67;
    r33 = fma(r33, r47, r57 * r69);
    r69 = r3 * r67;
    r69 = fma(r57, r69, r65 * r47);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r33, r69);
    r65 = r7 * r2;
    r73 = r18 * r73;
    r73 = r73 * r66;
    r75 = r66 * r75;
    r66 = r73 + r75;
    r55 = r7 * r22;
    r78 = r14 * r11;
    r68 = r8 * r15;
    r68 = fma(r63, r68, r59 * r78);
    r78 = r13 * r10;
    r68 = fma(r59, r78, r68);
    r68 = fma(r16, r64, r68);
    r55 = r55 * r68;
    r62 = r62 + r55;
    r64 = r18 * r24;
    r62 = fma(r12, r64, r62);
    r62 = r62 + r71;
    r62 = fma(r6, r62, r5 * r66);
    r66 = r7 * r18;
    r66 = r66 * r68;
    r71 = r7 * r23;
    r71 = fma(r77, r71, r66);
    r71 = r71 + r17;
    r62 = fma(r44, r71, r62);
    r71 = r40 * r62;
    r17 = r7 * r21;
    r64 = r7 * r23;
    r64 = fma(r68, r64, r12 * r17);
    r64 = r64 + r72;
    r17 = r22 * r24;
    r17 = fma(r12, r17, r80);
    r17 = r17 + r66;
    r17 = fma(r77, r25, r17);
    r17 = fma(r5, r17, r6 * r64);
    r79 = r68 * r79;
    r75 = r75 + r79;
    r17 = fma(r44, r75, r17);
    r71 = fma(r17, r49, r1 * r71);
    r75 = r17 * r48;
    r79 = r73 + r79;
    r73 = r21 * r24;
    r25 = fma(r68, r25, r12 * r73);
    r25 = r25 + r72;
    r25 = fma(r44, r25, r6 * r79);
    r55 = r19 + r55;
    r55 = r55 + r83;
    r25 = fma(r5, r55, r25);
    r55 = r0 * r25;
    r55 = fma(r1, r55, r34 * r75);
    r65 = fma(r55, r61, r71 * r65);
    r75 = r2 * r65;
    r71 = fma(r71, r47, r57 * r75);
    r75 = r3 * r65;
    r75 = fma(r57, r75, r55 * r47);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r71, r75);
    r55 = r7 * r2;
    r5 = r40 * r30;
    r5 = fma(r36, r49, r1 * r5);
    r83 = r0 * r28;
    r19 = r36 * r48;
    r19 = fma(r34, r19, r1 * r83);
    r55 = fma(r19, r61, r5 * r55);
    r83 = r2 * r55;
    r5 = fma(r5, r47, r57 * r83);
    r83 = r3 * r55;
    r19 = fma(r19, r47, r57 * r83);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r5, r19);
    r83 = r7 * r2;
    r44 = r40 * r35;
    r44 = fma(r37, r49, r1 * r44);
    r79 = r37 * r48;
    r6 = r0 * r20;
    r6 = fma(r1, r6, r34 * r79);
    r83 = fma(r6, r61, r44 * r83);
    r79 = r2 * r83;
    r44 = fma(r44, r47, r57 * r79);
    r79 = r3 * r83;
    r6 = fma(r6, r47, r57 * r79);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r44, r6);
    r79 = r7 * r2;
    r72 = r40 * r50;
    r72 = fma(r1, r72, r31 * r49);
    r68 = r0 * r51;
    r73 = r31 * r48;
    r73 = fma(r34, r73, r1 * r68);
    r79 = fma(r73, r61, r72 * r79);
    r68 = r2 * r79;
    r72 = fma(r72, r47, r57 * r68);
    r68 = r3 * r79;
    r68 = fma(r57, r68, r73 * r47);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r72, r68);
    r73 = r4 * r3;
    r73 = r73 * r60;
    r58 = r4 * r58;
    r73 = fma(r70, r58, r47 * r73);
    r12 = r4 * r3;
    r12 = r12 * r69;
    r12 = fma(r33, r58, r47 * r12);
    WriteSum2<double, double>((double*)inout_shared, r73, r12);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = r4 * r3;
    r12 = r12 * r75;
    r12 = fma(r47, r12, r71 * r58);
    r73 = r4 * r3;
    r73 = r73 * r19;
    r73 = fma(r5, r58, r47 * r73);
    WriteSum2<double, double>((double*)inout_shared, r12, r73);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r4 * r3;
    r73 = r73 * r6;
    r73 = fma(r47, r73, r44 * r58);
    r12 = r4 * r3;
    r12 = r12 * r68;
    r12 = fma(r47, r12, r72 * r58);
    WriteSum2<double, double>((double*)inout_shared, r73, r12);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r60, r60, r70 * r70);
    r73 = fma(r33, r33, r69 * r69);
    WriteSum2<double, double>((double*)inout_shared, r12, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r71, r71, r75 * r75);
    r12 = fma(r19, r19, r5 * r5);
    WriteSum2<double, double>((double*)inout_shared, r73, r12);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r6, r6, r44 * r44);
    r73 = fma(r72, r72, r68 * r68);
    WriteSum2<double, double>((double*)inout_shared, r12, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r70, r33, r60 * r69);
    r12 = fma(r60, r75, r70 * r71);
    WriteSum2<double, double>((double*)inout_shared, r73, r12);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r70, r5, r60 * r19);
    r73 = fma(r70, r44, r60 * r6);
    WriteSum2<double, double>((double*)inout_shared, r12, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r70, r72, r60 * r68);
    r60 = fma(r33, r71, r69 * r75);
    WriteSum2<double, double>((double*)inout_shared, r70, r60);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r33, r5, r69 * r19);
    r70 = fma(r33, r44, r69 * r6);
    WriteSum2<double, double>((double*)inout_shared, r60, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r33, r72, r69 * r68);
    r69 = fma(r71, r5, r75 * r19);
    WriteSum2<double, double>((double*)inout_shared, r33, r69);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fma(r75, r6, r71 * r44);
    r75 = fma(r75, r68, r71 * r72);
    WriteSum2<double, double>((double*)inout_shared, r69, r75);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fma(r5, r44, r19 * r6);
    r5 = fma(r5, r72, r19 * r68);
    WriteSum2<double, double>((double*)inout_shared, r75, r5);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r44, r72, r6 * r68);
    WriteSum1<double, double>((double*)inout_shared, r72);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r52 * r40;
    r72 = r72 * r41;
    r72 = fma(r1, r72, r49 * r39);
    r44 = r7 * r2;
    r68 = r53 * r0;
    r68 = r68 * r41;
    r41 = r48 * r34;
    r41 = fma(r39, r41, r1 * r68);
    r44 = fma(r41, r61, r72 * r44);
    r68 = r2 * r44;
    r68 = fma(r57, r68, r72 * r47);
    r72 = r3 * r44;
    r41 = fma(r41, r47, r57 * r72);
    WriteIdx2<1024, double, double, double2>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r68,
        r41);
    r72 = r4 * r3;
    r72 = r72 * r41;
    r72 = fma(r68, r58, r47 * r72);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_njtr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r72);
  if (global_thread_idx < problem_size) {
    r68 = fma(r68, r68, r41 * r41);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_precond_diag_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r68);
  if (global_thread_idx < problem_size) {
    r68 = r40 * r45;
    r68 = fma(r26, r49, r1 * r68);
    r41 = r7 * r2;
    r72 = r26 * r48;
    r39 = r0 * r38;
    r39 = fma(r1, r39, r34 * r72);
    r41 = fma(r39, r61, r68 * r41);
    r72 = r2 * r41;
    r72 = fma(r57, r72, r68 * r47);
    r68 = r3 * r41;
    r68 = fma(r57, r68, r39 * r47);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r72,
                                             r68);
    r39 = r40 * r56;
    r39 = fma(r43, r49, r1 * r39);
    r6 = r7 * r2;
    r5 = r0 * r54;
    r75 = r43 * r48;
    r75 = fma(r34, r75, r1 * r5);
    r6 = fma(r75, r61, r39 * r6);
    r5 = r2 * r6;
    r5 = fma(r57, r5, r39 * r47);
    r39 = r3 * r6;
    r75 = fma(r75, r47, r57 * r39);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r5, r75);
    r39 = r40 * r32;
    r49 = fma(r46, r49, r1 * r39);
    r39 = r7 * r2;
    r19 = r46 * r48;
    r69 = r0 * r42;
    r69 = fma(r1, r69, r34 * r19);
    r61 = fma(r69, r61, r49 * r39);
    r39 = r2 * r61;
    r39 = fma(r57, r39, r49 * r47);
    r49 = r3 * r61;
    r69 = fma(r69, r47, r57 * r49);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r39,
                                             r69);
    r49 = r4 * r3;
    r49 = r49 * r68;
    r49 = fma(r72, r58, r47 * r49);
    r57 = r4 * r3;
    r57 = r57 * r75;
    r57 = fma(r47, r57, r5 * r58);
    WriteSum2<double, double>((double*)inout_shared, r49, r57);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r4 * r3;
    r57 = r57 * r69;
    r57 = fma(r47, r57, r39 * r58);
    WriteSum1<double, double>((double*)inout_shared, r57);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r72, r72, r68 * r68);
    r58 = fma(r5, r5, r75 * r75);
    WriteSum2<double, double>((double*)inout_shared, r57, r58);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fma(r69, r69, r39 * r39);
    WriteSum1<double, double>((double*)inout_shared, r58);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fma(r72, r5, r68 * r75);
    r72 = fma(r72, r39, r68 * r69);
    WriteSum2<double, double>((double*)inout_shared, r58, r72);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fma(r75, r69, r5 * r39);
    WriteSum1<double, double>((double*)inout_shared, r69);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
  SumFlushFinal<double>(out_sensor_from_rig_log_scale_njtr_local,
                        out_sensor_from_rig_log_scale_njtr,
                        1);
  SumFlushFinal<double>(out_sensor_from_rig_log_scale_precond_diag_local,
                        out_sensor_from_rig_log_scale_precond_diag,
                        1);
}

void RowFixedRigPinholeResJacFirst(
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
    double* const out_rTr,
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
  RowFixedRigPinholeResJacFirstKernel<<<n_blocks, 1024>>>(
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
      out_rTr,
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