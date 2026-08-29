#include "kernel_fixed_rig_pinhole_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) FixedRigPinholeResJacKernel(
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
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib, 2 * calib_num_alloc, global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    r0 = 1.00000000000000008e-15;
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
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r10,
                                            r11);
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r14,
                                            r15);
    r16 = r13 * r14;
    r17 = fma(r8, r11, r16);
    r18 = r9 * r10;
    r19 = r12 * r15;
    r17 = r17 + r18;
    r17 = fma(r4, r19, r17);
    r20 = r7 * r17;
    r21 = fma(r9, r14, r12 * r11);
    r22 = r13 * r10;
    r21 = fma(r4, r22, r21);
    r21 = fma(r8, r15, r21);
    r20 = r20 * r21;
    r22 = r8 * r14;
    r22 = fma(r4, r22, r13 * r11);
    r22 = fma(r9, r15, r22);
    r22 = fma(r12, r10, r22);
    r23 = -2.00000000000000000e+00;
    r24 = fma(r13, r15, r12 * r14);
    r24 = fma(r8, r10, r24);
    r24 = fma(r4, r24, r9 * r11);
    r25 = r23 * r24;
    r26 = fma(r22, r25, r20);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r27, r28);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r29 = r14 * r10;
    r29 = r29 * r7;
    r30 = r15 * r11;
    r30 = fma(r23, r30, r29);
    r31 = fma(r27, r30, r5 * r26);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r14 * r14;
    r33 = r33 * r23;
    r34 = 1.00000000000000000e+00;
    r35 = r15 * r15;
    r35 = fma(r23, r35, r34);
    r36 = r33 + r35;
    r37 = r15 * r10;
    r37 = r37 * r7;
    r38 = r14 * r11;
    r38 = fma(r7, r38, r37);
    r39 = r7 * r17;
    r39 = r39 * r22;
    r40 = r7 * r21;
    r40 = fma(r24, r40, r39);
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r41);
  };
  LoadUnique<1, double, double>(
      sensor_from_rig_log_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = 2.71828182845904523536;
    r42 = pow(r43, r42);
    r41 = r41 * r42;
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r43);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r44 = r23 * r22;
    r44 = r44 * r22;
    r45 = r34 + r44;
    r46 = r23 * r21;
    r46 = r46 * r21;
    r45 = r45 + r46;
    r31 = fma(r32, r36, r31);
    r31 = fma(r28, r38, r31);
    r31 = fma(r6, r40, r31);
    r31 = r31 + r41;
    r31 = fma(r43, r45, r31);
    r47 = copysign(1.0, r31);
    r47 = fma(r0, r47, r31);
    r0 = 1.0 / r47;
    ReadIdx2<1024, double, double, double2>(
        calib, 0 * calib_num_alloc, global_thread_idx, r31, r48);
    r49 = r23 * r17;
    r49 = r49 * r17;
    r50 = r34 + r49;
    r50 = r50 + r44;
    r44 = r7 * r22;
    r44 = r44 * r21;
    r51 = fma(r17, r25, r44);
    r52 = fma(r6, r51, r5 * r50);
    r53 = r7 * r22;
    r53 = fma(r24, r53, r20);
    r20 = r15 * r11;
    r20 = fma(r7, r20, r29);
    r29 = r10 * r11;
    r54 = r14 * r15;
    r54 = r54 * r7;
    r29 = fma(r23, r29, r54);
    r55 = r10 * r10;
    r55 = r55 * r23;
    r35 = r55 + r35;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r56,
                                            r57);
    r52 = fma(r43, r53, r52);
    r52 = fma(r32, r20, r52);
    r52 = fma(r28, r29, r52);
    r52 = fma(r27, r35, r52);
    r52 = fma(r56, r42, r52);
    r52 = r31 * r52;
    r2 = fma(r0, r52, r2);
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
    r3 = fma(r3, r4, r1);
    r1 = r7 * r17;
    r1 = fma(r24, r1, r44);
    r44 = r10 * r11;
    r44 = fma(r7, r44, r54);
    r27 = fma(r27, r44, r5 * r1);
    r55 = r34 + r55;
    r55 = r55 + r33;
    r33 = r14 * r11;
    r33 = fma(r23, r33, r37);
    r39 = fma(r21, r25, r39);
    r49 = r34 + r49;
    r49 = r49 + r46;
    r27 = fma(r28, r55, r27);
    r27 = fma(r32, r33, r27);
    r27 = fma(r43, r39, r27);
    r27 = fma(r57, r42, r27);
    r27 = fma(r6, r49, r27);
    r27 = r48 * r27;
    r3 = fma(r0, r27, r3);
    r32 = fma(r3, r3, r2 * r2);
    r32 = fma(r32, r58, r34);
    r28 = sqrt(r32);
    r28 = r34 + r28;
    r34 = 1.0 / r28;
    r34 = r7 * r34;
    r46 = sqrt(r34);
    r37 = r2 * r46;
    r54 = r3 * r46;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r37, r54);
    r54 = r7 * r2;
    r59 = r7 * r24;
    r60 = 5.00000000000000000e-01;
    r61 = r11 * r60;
    r62 = fma(r60, r16, r8 * r61);
    r63 = -5.00000000000000000e-01;
    r62 = fma(r63, r19, r62);
    r62 = fma(r60, r18, r62);
    r64 = r12 * r11;
    r65 = r9 * r14;
    r65 = fma(r63, r65, r63 * r64);
    r64 = r8 * r15;
    r65 = fma(r63, r64, r65);
    r66 = r13 * r10;
    r65 = fma(r60, r66, r65);
    r66 = r22 * r65;
    r59 = fma(r7, r66, r62 * r59);
    r64 = r7 * r21;
    r67 = r13 * r11;
    r68 = r8 * r14;
    r68 = fma(r60, r68, r63 * r67);
    r67 = r9 * r15;
    r68 = fma(r63, r67, r68);
    r69 = r12 * r10;
    r68 = fma(r63, r69, r68);
    r69 = r7 * r17;
    r67 = r12 * r14;
    r70 = r13 * r15;
    r70 = fma(r63, r70, r63 * r67);
    r67 = r8 * r10;
    r70 = fma(r63, r67, r70);
    r70 = fma(r9, r61, r70);
    r69 = r69 * r70;
    r64 = fma(r68, r64, r69);
    r59 = r59 + r64;
    r67 = r7 * r22;
    r67 = r67 * r70;
    r71 = r7 * r21;
    r71 = r71 * r62;
    r72 = r67 + r71;
    r73 = r23 * r17;
    r72 = fma(r65, r73, r72);
    r72 = fma(r68, r25, r72);
    r72 = fma(r6, r72, r43 * r59);
    r59 = r22 * r62;
    r73 = -4.00000000000000000e+00;
    r59 = r59 * r73;
    r74 = r17 * r68;
    r75 = r73 * r74;
    r76 = r59 + r75;
    r72 = fma(r5, r76, r72);
    r76 = r31 * r72;
    r77 = r7 * r22;
    r77 = r77 * r68;
    r78 = r7 * r17;
    r78 = fma(r62, r78, r77);
    r79 = r7 * r21;
    r79 = r79 * r65;
    r80 = r7 * r24;
    r80 = r80 * r70;
    r81 = r79 + r80;
    r82 = r78 + r81;
    r62 = fma(r62, r25, r23 * r66);
    r62 = r62 + r64;
    r62 = fma(r5, r62, r6 * r82);
    r82 = r21 * r73;
    r83 = r70 * r82;
    r59 = r59 + r83;
    r62 = fma(r43, r59, r62);
    r47 = r47 * r47;
    r47 = 1.0 / r47;
    r47 = r4 * r47;
    r52 = r47 * r52;
    r76 = fma(r62, r52, r0 * r76);
    r59 = r7 * r3;
    r84 = r23 * r21;
    r85 = r70 * r25;
    r84 = fma(r65, r84, r85);
    r84 = r84 + r78;
    r75 = r83 + r75;
    r75 = fma(r6, r75, r43 * r84);
    r84 = r7 * r24;
    r84 = fma(r68, r84, r71);
    r71 = r7 * r17;
    r71 = fma(r65, r71, r67);
    r84 = r84 + r71;
    r75 = fma(r5, r84, r75);
    r84 = r48 * r75;
    r67 = r62 * r47;
    r67 = fma(r27, r67, r0 * r84);
    r54 = fma(r67, r59, r76 * r54);
    r58 = r63 * r58;
    r34 = rsqrt(r34);
    r28 = r28 * r28;
    r28 = 1.0 / r28;
    r32 = rsqrt(r32);
    r58 = r58 * r34;
    r58 = r58 * r28;
    r58 = r58 * r32;
    r54 = r54 * r58;
    r76 = fma(r76, r46, r2 * r54);
    r67 = fma(r67, r46, r3 * r54);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r76, r67);
    r54 = r23 * r22;
    r54 = fma(r68, r54, r85);
    r32 = r7 * r17;
    r28 = r8 * r11;
    r16 = fma(r63, r16, r63 * r28);
    r16 = fma(r60, r19, r16);
    r16 = fma(r63, r18, r16);
    r32 = r32 * r16;
    r18 = r7 * r21;
    r19 = r9 * r14;
    r28 = r8 * r15;
    r28 = fma(r60, r28, r60 * r19);
    r19 = r13 * r10;
    r28 = fma(r63, r19, r28);
    r28 = fma(r12, r61, r28);
    r18 = fma(r28, r18, r32);
    r54 = r54 + r18;
    r19 = r7 * r22;
    r19 = r19 * r28;
    r34 = r7 * r24;
    r34 = fma(r16, r34, r19);
    r34 = r34 + r64;
    r34 = fma(r6, r34, r5 * r54);
    r54 = r22 * r70;
    r54 = r54 * r73;
    r64 = r16 * r82;
    r84 = r54 + r64;
    r34 = fma(r43, r84, r34);
    r80 = r77 + r80;
    r80 = r80 + r18;
    r18 = r17 * r73;
    r18 = r18 * r28;
    r54 = r54 + r18;
    r54 = fma(r5, r54, r43 * r80);
    r80 = fma(r28, r25, r23 * r74);
    r77 = r7 * r21;
    r77 = r77 * r70;
    r84 = r7 * r22;
    r84 = fma(r16, r84, r77);
    r80 = r80 + r84;
    r54 = fma(r6, r80, r54);
    r80 = r31 * r54;
    r80 = fma(r0, r80, r34 * r52);
    r83 = r7 * r2;
    r19 = r69 + r19;
    r69 = r23 * r21;
    r19 = fma(r68, r69, r19);
    r19 = fma(r16, r25, r19);
    r69 = r7 * r24;
    r74 = fma(r7, r74, r28 * r69);
    r74 = r74 + r84;
    r74 = fma(r5, r74, r43 * r19);
    r64 = r18 + r64;
    r74 = fma(r6, r64, r74);
    r64 = r48 * r74;
    r18 = r34 * r47;
    r18 = fma(r27, r18, r0 * r64);
    r83 = fma(r18, r59, r80 * r83);
    r64 = r2 * r83;
    r64 = fma(r58, r64, r80 * r46);
    r80 = r3 * r83;
    r80 = fma(r58, r80, r18 * r46);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r64, r80);
    r18 = r7 * r2;
    r19 = r7 * r22;
    r69 = r8 * r14;
    r28 = r9 * r15;
    r28 = fma(r60, r28, r63 * r69);
    r69 = r12 * r10;
    r28 = fma(r60, r69, r28);
    r28 = fma(r13, r61, r28);
    r19 = r19 * r28;
    r79 = r79 + r19;
    r61 = r23 * r17;
    r79 = fma(r16, r61, r79);
    r79 = r79 + r85;
    r70 = r17 * r70;
    r70 = r70 * r73;
    r66 = r73 * r66;
    r73 = r70 + r66;
    r73 = fma(r5, r73, r6 * r79);
    r79 = r7 * r17;
    r79 = r79 * r28;
    r85 = r7 * r24;
    r85 = fma(r65, r85, r79);
    r85 = r85 + r84;
    r73 = fma(r43, r85, r73);
    r85 = r31 * r73;
    r82 = r28 * r82;
    r66 = r82 + r66;
    r79 = r77 + r79;
    r77 = r23 * r22;
    r79 = fma(r16, r77, r79);
    r79 = fma(r65, r25, r79);
    r79 = fma(r5, r79, r43 * r66);
    r66 = r7 * r21;
    r65 = r7 * r24;
    r65 = fma(r28, r65, r16 * r66);
    r65 = r65 + r71;
    r79 = fma(r6, r65, r79);
    r85 = fma(r79, r52, r0 * r85);
    r65 = r79 * r47;
    r19 = r32 + r19;
    r19 = r19 + r81;
    r81 = r23 * r21;
    r25 = fma(r28, r25, r16 * r81);
    r25 = r25 + r71;
    r25 = fma(r43, r25, r5 * r19);
    r82 = r70 + r82;
    r25 = fma(r6, r82, r25);
    r82 = r48 * r25;
    r82 = fma(r0, r82, r27 * r65);
    r18 = fma(r82, r59, r85 * r18);
    r65 = r2 * r18;
    r85 = fma(r85, r46, r58 * r65);
    r65 = r3 * r18;
    r82 = fma(r82, r46, r58 * r65);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r85, r82);
    r65 = r31 * r35;
    r65 = fma(r0, r65, r30 * r52);
    r6 = r7 * r2;
    r70 = r30 * r47;
    r43 = r48 * r44;
    r43 = fma(r0, r43, r27 * r70);
    r6 = fma(r43, r59, r65 * r6);
    r70 = r2 * r6;
    r70 = fma(r58, r70, r65 * r46);
    r65 = r3 * r6;
    r43 = fma(r43, r46, r58 * r65);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r70, r43);
    r65 = r31 * r29;
    r65 = fma(r0, r65, r38 * r52);
    r19 = r7 * r2;
    r5 = r48 * r55;
    r71 = r38 * r47;
    r71 = fma(r27, r71, r0 * r5);
    r19 = fma(r71, r59, r65 * r19);
    r5 = r2 * r19;
    r5 = fma(r58, r5, r65 * r46);
    r65 = r3 * r19;
    r71 = fma(r71, r46, r58 * r65);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r5, r71);
    r65 = r31 * r20;
    r65 = fma(r0, r65, r36 * r52);
    r28 = r7 * r2;
    r81 = r36 * r47;
    r16 = r48 * r33;
    r16 = fma(r0, r16, r27 * r81);
    r28 = fma(r16, r59, r65 * r28);
    r81 = r2 * r28;
    r81 = fma(r58, r81, r65 * r46);
    r65 = r3 * r28;
    r65 = fma(r58, r65, r16 * r46);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r81, r65);
    r37 = r4 * r37;
    r16 = r4 * r3;
    r16 = r16 * r67;
    r16 = fma(r46, r16, r76 * r37);
    r32 = r4 * r3;
    r32 = r32 * r80;
    r32 = fma(r46, r32, r64 * r37);
    WriteSum2<double, double>((double*)inout_shared, r16, r32);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r4 * r3;
    r32 = r32 * r82;
    r32 = fma(r46, r32, r85 * r37);
    r16 = r4 * r3;
    r16 = r16 * r43;
    r16 = fma(r46, r16, r70 * r37);
    WriteSum2<double, double>((double*)inout_shared, r32, r16);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r4 * r3;
    r16 = r16 * r71;
    r16 = fma(r46, r16, r5 * r37);
    r32 = r4 * r3;
    r32 = r32 * r65;
    r32 = fma(r46, r32, r81 * r37);
    WriteSum2<double, double>((double*)inout_shared, r16, r32);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r67, r67, r76 * r76);
    r16 = fma(r80, r80, r64 * r64);
    WriteSum2<double, double>((double*)inout_shared, r32, r16);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fma(r82, r82, r85 * r85);
    r32 = fma(r43, r43, r70 * r70);
    WriteSum2<double, double>((double*)inout_shared, r16, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r5, r5, r71 * r71);
    r16 = fma(r65, r65, r81 * r81);
    WriteSum2<double, double>((double*)inout_shared, r32, r16);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fma(r76, r64, r67 * r80);
    r32 = fma(r67, r82, r76 * r85);
    WriteSum2<double, double>((double*)inout_shared, r16, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r76, r70, r67 * r43);
    r16 = fma(r67, r71, r76 * r5);
    WriteSum2<double, double>((double*)inout_shared, r32, r16);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fma(r76, r81, r67 * r65);
    r67 = fma(r64, r85, r80 * r82);
    WriteSum2<double, double>((double*)inout_shared, r76, r67);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = fma(r64, r70, r80 * r43);
    r76 = fma(r80, r71, r64 * r5);
    WriteSum2<double, double>((double*)inout_shared, r67, r76);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r80, r65, r64 * r81);
    r64 = fma(r85, r70, r82 * r43);
    WriteSum2<double, double>((double*)inout_shared, r80, r64);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fma(r82, r71, r85 * r5);
    r85 = fma(r85, r81, r82 * r65);
    WriteSum2<double, double>((double*)inout_shared, r64, r85);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = fma(r43, r71, r70 * r5);
    r70 = fma(r70, r81, r43 * r65);
    WriteSum2<double, double>((double*)inout_shared, r85, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fma(r71, r65, r5 * r81);
    WriteSum1<double, double>((double*)inout_shared, r65);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r56 * r31;
    r65 = r65 * r42;
    r65 = fma(r0, r65, r52 * r41);
    r71 = r7 * r2;
    r81 = r47 * r27;
    r5 = r57 * r48;
    r5 = r5 * r42;
    r5 = fma(r0, r5, r41 * r81);
    r71 = fma(r5, r59, r65 * r71);
    r81 = r2 * r71;
    r81 = fma(r58, r81, r65 * r46);
    r65 = r3 * r71;
    r65 = fma(r58, r65, r5 * r46);
    WriteIdx2<1024, double, double, double2>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r81,
        r65);
    r5 = r4 * r3;
    r5 = r5 * r65;
    r5 = fma(r81, r37, r46 * r5);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_njtr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r5);
  if (global_thread_idx < problem_size) {
    r81 = fma(r81, r81, r65 * r65);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_precond_diag_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r81);
  if (global_thread_idx < problem_size) {
    r81 = r7 * r2;
    r65 = r31 * r50;
    r65 = fma(r0, r65, r26 * r52);
    r5 = r26 * r47;
    r41 = r48 * r1;
    r41 = fma(r0, r41, r27 * r5);
    r81 = fma(r41, r59, r65 * r81);
    r5 = r2 * r81;
    r65 = fma(r65, r46, r58 * r5);
    r5 = r3 * r81;
    r5 = fma(r58, r5, r41 * r46);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r65, r5);
    r41 = r31 * r51;
    r41 = fma(r40, r52, r0 * r41);
    r42 = r7 * r2;
    r70 = r40 * r47;
    r85 = r48 * r49;
    r85 = fma(r0, r85, r27 * r70);
    r42 = fma(r85, r59, r41 * r42);
    r70 = r2 * r42;
    r70 = fma(r58, r70, r41 * r46);
    r41 = r3 * r42;
    r41 = fma(r58, r41, r85 * r46);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r70,
                                             r41);
    r85 = r7 * r2;
    r43 = r31 * r53;
    r43 = fma(r0, r43, r45 * r52);
    r52 = r48 * r39;
    r64 = r45 * r47;
    r64 = fma(r27, r64, r0 * r52);
    r59 = fma(r64, r59, r43 * r85);
    r85 = r2 * r59;
    r43 = fma(r43, r46, r58 * r85);
    r85 = r3 * r59;
    r85 = fma(r58, r85, r64 * r46);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r43,
                                             r85);
    r64 = r4 * r3;
    r64 = r64 * r5;
    r64 = fma(r65, r37, r46 * r64);
    r58 = r4 * r3;
    r58 = r58 * r41;
    r58 = fma(r46, r58, r70 * r37);
    WriteSum2<double, double>((double*)inout_shared, r64, r58);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = r4 * r3;
    r58 = r58 * r85;
    r37 = fma(r43, r37, r46 * r58);
    WriteSum1<double, double>((double*)inout_shared, r37);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r65, r65, r5 * r5);
    r58 = fma(r70, r70, r41 * r41);
    WriteSum2<double, double>((double*)inout_shared, r37, r58);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fma(r85, r85, r43 * r43);
    WriteSum1<double, double>((double*)inout_shared, r58);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fma(r5, r41, r65 * r70);
    r65 = fma(r65, r43, r5 * r85);
    WriteSum2<double, double>((double*)inout_shared, r58, r65);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fma(r70, r43, r41 * r85);
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
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedRigPinholeResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      sensor_from_rig_log_scale,
      calib,
      calib_num_alloc,
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