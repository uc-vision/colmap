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
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    const float* const sensor_from_rig_log_scale,
    float* calib,
    unsigned int calib_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_sensor_from_rig_log_scale_jac,
    unsigned int out_sensor_from_rig_log_scale_jac_num_alloc,
    float* const out_sensor_from_rig_log_scale_njtr,
    float* const out_sensor_from_rig_log_scale_precond_diag,
    float* const out_sensor_from_rig_log_scale_precond_tril,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
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

  __shared__ float out_sensor_from_rig_log_scale_njtr_local[1];

  __shared__ float out_sensor_from_rig_log_scale_precond_diag_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        calib, 0 * calib_num_alloc, global_thread_idx, r0, r1, r2, r3);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r2);
    r2 = 9.99999999999999955e-07;
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r7,
                       r8,
                       r9);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r10,
                       r11,
                       r12,
                       r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r15,
                                         r16,
                                         r17);
    r18 = r12 * r17;
    r19 = fmaf(r11, r14, r18);
    r20 = r10 * r15;
    r19 = fmaf(r13, r16, r19);
    r19 = fmaf(r6, r20, r19);
    r21 = 2.00000000000000000e+00;
    r22 = r19 * r21;
    r23 = fmaf(r13, r14, r10 * r17);
    r24 = r11 * r16;
    r23 = fmaf(r6, r24, r23);
    r23 = fmaf(r12, r15, r23);
    r22 = r22 * r23;
    r24 = r12 * r14;
    r24 = fmaf(r6, r24, r11 * r17);
    r24 = fmaf(r13, r15, r24);
    r24 = fmaf(r10, r16, r24);
    r25 = -2.00000000000000000e+00;
    r26 = fmaf(r11, r15, r10 * r14);
    r26 = fmaf(r12, r16, r26);
    r26 = fmaf(r6, r26, r13 * r17);
    r27 = r25 * r26;
    r28 = fmaf(r24, r27, r22);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r29,
                       r30,
                       r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r14 * r16;
    r32 = r32 * r21;
    r33 = r15 * r17;
    r33 = fmaf(r25, r33, r32);
    r34 = fmaf(r29, r33, r7 * r28);
    r35 = r14 * r14;
    r35 = r25 * r35;
    r36 = 1.00000000000000000e+00;
    r37 = r15 * r15;
    r37 = fmaf(r25, r37, r36);
    r38 = r35 + r37;
    r39 = r15 * r16;
    r39 = r39 * r21;
    r40 = r14 * r17;
    r40 = fmaf(r21, r40, r39);
    r41 = r21 * r23;
    r42 = r21 * r24;
    r43 = r19 * r42;
    r41 = fmaf(r26, r41, r43);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r44,
                                         r45,
                                         r46);
  };
  LoadUnique<1, float, float>(
      sensor_from_rig_log_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r47);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r48 = 2.71828182845904523536;
    r47 = powf(r48, r47);
    r46 = r46 * r47;
    r48 = r25 * r24;
    r48 = r48 * r24;
    r49 = r36 + r48;
    r50 = r25 * r23;
    r50 = r50 * r23;
    r49 = r49 + r50;
    r34 = fmaf(r31, r38, r34);
    r34 = fmaf(r30, r40, r34);
    r34 = fmaf(r8, r41, r34);
    r34 = r34 + r46;
    r34 = fmaf(r9, r49, r34);
    r51 = copysign(1.0, r34);
    r51 = fmaf(r2, r51, r34);
    r2 = 1.0 / r51;
    r34 = r25 * r19;
    r34 = r34 * r19;
    r52 = r36 + r34;
    r52 = r52 + r48;
    r48 = r23 * r42;
    r53 = fmaf(r19, r27, r48);
    r54 = fmaf(r8, r53, r7 * r52);
    r22 = fmaf(r26, r42, r22);
    r55 = r15 * r17;
    r55 = fmaf(r21, r55, r32);
    r32 = r16 * r17;
    r56 = r14 * r15;
    r56 = r56 * r21;
    r32 = fmaf(r25, r32, r56);
    r57 = r16 * r16;
    r57 = r25 * r57;
    r37 = r57 + r37;
    r54 = fmaf(r9, r22, r54);
    r54 = fmaf(r31, r55, r54);
    r54 = fmaf(r30, r32, r54);
    r54 = fmaf(r29, r37, r54);
    r54 = fmaf(r44, r47, r54);
    r54 = r0 * r54;
    r4 = fmaf(r2, r54, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r19 * r21;
    r3 = fmaf(r26, r3, r48);
    r48 = r16 * r17;
    r48 = fmaf(r21, r48, r56);
    r29 = fmaf(r29, r48, r7 * r3);
    r57 = r36 + r57;
    r57 = r57 + r35;
    r35 = r14 * r17;
    r35 = fmaf(r25, r35, r39);
    r43 = fmaf(r23, r27, r43);
    r34 = r36 + r34;
    r34 = r34 + r50;
    r29 = fmaf(r30, r57, r29);
    r29 = fmaf(r31, r35, r29);
    r29 = fmaf(r9, r43, r29);
    r29 = fmaf(r45, r47, r29);
    r29 = fmaf(r8, r34, r29);
    r29 = r1 * r29;
    r5 = fmaf(r2, r29, r5);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r31 = r21 * r26;
    r30 = r11 * r14;
    r50 = 5.00000000000000000e-01;
    r30 = fmaf(r50, r18, r50 * r30);
    r36 = -5.00000000000000000e-01;
    r39 = r13 * r50;
    r30 = fmaf(r36, r20, r30);
    r30 = fmaf(r16, r39, r30);
    r56 = r10 * r17;
    r58 = r13 * r14;
    r58 = fmaf(r36, r58, r36 * r56);
    r56 = r12 * r15;
    r58 = fmaf(r36, r56, r58);
    r59 = r11 * r16;
    r58 = fmaf(r50, r59, r58);
    r31 = fmaf(r58, r42, r30 * r31);
    r59 = r21 * r23;
    r56 = r12 * r14;
    r60 = r13 * r15;
    r60 = fmaf(r36, r60, r50 * r56);
    r56 = r10 * r16;
    r60 = fmaf(r36, r56, r60);
    r61 = r11 * r36;
    r60 = fmaf(r17, r61, r60);
    r56 = r19 * r21;
    r62 = r10 * r14;
    r63 = r12 * r16;
    r63 = fmaf(r36, r63, r36 * r62);
    r63 = fmaf(r17, r39, r63);
    r63 = fmaf(r15, r61, r63);
    r56 = r56 * r63;
    r59 = fmaf(r60, r59, r56);
    r31 = r31 + r59;
    r62 = r21 * r23;
    r62 = r62 * r30;
    r64 = r25 * r19;
    r64 = fmaf(r58, r64, r62);
    r65 = r63 * r42;
    r64 = r64 + r65;
    r64 = fmaf(r60, r27, r64);
    r64 = fmaf(r8, r64, r9 * r31);
    r31 = r24 * r30;
    r66 = -4.00000000000000000e+00;
    r31 = r31 * r66;
    r67 = r19 * r60;
    r68 = r66 * r67;
    r69 = r31 + r68;
    r64 = fmaf(r7, r69, r64);
    r69 = r0 * r64;
    r70 = r19 * r21;
    r71 = r60 * r42;
    r70 = fmaf(r30, r70, r71);
    r72 = r21 * r23;
    r72 = r72 * r58;
    r73 = r21 * r26;
    r73 = r73 * r63;
    r74 = r72 + r73;
    r75 = r70 + r74;
    r76 = r25 * r24;
    r30 = fmaf(r30, r27, r58 * r76);
    r30 = r30 + r59;
    r30 = fmaf(r7, r30, r8 * r75);
    r75 = r23 * r66;
    r76 = r63 * r75;
    r31 = r31 + r76;
    r30 = fmaf(r9, r31, r30);
    r51 = r51 * r51;
    r51 = 1.0 / r51;
    r51 = r6 * r51;
    r54 = r51 * r54;
    r69 = fmaf(r30, r54, r2 * r69);
    r31 = r30 * r51;
    r77 = r25 * r23;
    r78 = r63 * r27;
    r77 = fmaf(r58, r77, r78);
    r77 = r77 + r70;
    r76 = r68 + r76;
    r76 = fmaf(r8, r76, r9 * r77);
    r77 = r21 * r26;
    r77 = fmaf(r60, r77, r62);
    r62 = r19 * r21;
    r62 = fmaf(r58, r62, r65);
    r77 = r77 + r62;
    r76 = fmaf(r7, r77, r76);
    r77 = r1 * r76;
    r77 = fmaf(r2, r77, r29 * r31);
    r31 = r25 * r24;
    r31 = fmaf(r60, r31, r78);
    r65 = r19 * r21;
    r68 = r13 * r16;
    r18 = fmaf(r36, r18, r36 * r68);
    r18 = fmaf(r14, r61, r18);
    r18 = fmaf(r50, r20, r18);
    r65 = r65 * r18;
    r20 = r21 * r23;
    r68 = r10 * r17;
    r70 = r12 * r15;
    r70 = fmaf(r50, r70, r50 * r68);
    r70 = fmaf(r14, r39, r70);
    r70 = fmaf(r16, r61, r70);
    r20 = fmaf(r70, r20, r65);
    r31 = r31 + r20;
    r61 = r21 * r26;
    r68 = r70 * r42;
    r61 = fmaf(r18, r61, r68);
    r61 = r61 + r59;
    r61 = fmaf(r8, r61, r7 * r31);
    r31 = r24 * r63;
    r31 = r31 * r66;
    r59 = r18 * r75;
    r79 = r31 + r59;
    r61 = fmaf(r9, r79, r61);
    r71 = r73 + r71;
    r71 = r71 + r20;
    r20 = r19 * r66;
    r20 = r20 * r70;
    r31 = r31 + r20;
    r31 = fmaf(r7, r31, r9 * r71);
    r71 = fmaf(r70, r27, r25 * r67);
    r73 = r21 * r23;
    r73 = r73 * r63;
    r79 = fmaf(r18, r42, r73);
    r71 = r71 + r79;
    r31 = fmaf(r8, r71, r31);
    r71 = r0 * r31;
    r71 = fmaf(r2, r71, r61 * r54);
    r80 = r25 * r23;
    r80 = fmaf(r60, r80, r56);
    r80 = r80 + r68;
    r80 = fmaf(r18, r27, r80);
    r68 = r21 * r26;
    r67 = fmaf(r21, r67, r70 * r68);
    r67 = r67 + r79;
    r67 = fmaf(r7, r67, r9 * r80);
    r59 = r20 + r59;
    r67 = fmaf(r8, r59, r67);
    r59 = r1 * r67;
    r20 = r61 * r51;
    r20 = fmaf(r29, r20, r2 * r59);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r69,
                                          r77,
                                          r71,
                                          r20);
    r59 = r25 * r19;
    r59 = fmaf(r18, r59, r72);
    r72 = r11 * r17;
    r80 = r12 * r14;
    r80 = fmaf(r36, r80, r50 * r72);
    r72 = r10 * r16;
    r80 = fmaf(r50, r72, r80);
    r80 = fmaf(r15, r39, r80);
    r42 = r80 * r42;
    r59 = r59 + r78;
    r59 = r59 + r42;
    r63 = r19 * r63;
    r63 = r63 * r66;
    r78 = r24 * r58;
    r78 = r78 * r66;
    r66 = r63 + r78;
    r66 = fmaf(r7, r66, r8 * r59);
    r59 = r19 * r21;
    r59 = r59 * r80;
    r39 = r21 * r26;
    r39 = fmaf(r58, r39, r59);
    r39 = r39 + r79;
    r66 = fmaf(r9, r39, r66);
    r39 = r0 * r66;
    r75 = r80 * r75;
    r78 = r78 + r75;
    r59 = r73 + r59;
    r73 = r25 * r24;
    r59 = fmaf(r18, r73, r59);
    r59 = fmaf(r58, r27, r59);
    r59 = fmaf(r7, r59, r9 * r78);
    r78 = r21 * r23;
    r58 = r21 * r26;
    r58 = fmaf(r80, r58, r18 * r78);
    r58 = r58 + r62;
    r59 = fmaf(r8, r58, r59);
    r39 = fmaf(r59, r54, r2 * r39);
    r58 = r59 * r51;
    r42 = r65 + r42;
    r42 = r42 + r74;
    r74 = r25 * r23;
    r27 = fmaf(r80, r27, r18 * r74);
    r27 = r27 + r62;
    r27 = fmaf(r9, r27, r7 * r42);
    r75 = r63 + r75;
    r27 = fmaf(r8, r75, r27);
    r75 = r1 * r27;
    r75 = fmaf(r2, r75, r29 * r58);
    r58 = r0 * r37;
    r58 = fmaf(r2, r58, r33 * r54);
    r8 = r33 * r51;
    r63 = r1 * r48;
    r63 = fmaf(r2, r63, r29 * r8);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r39,
                                          r75,
                                          r58,
                                          r63);
    r8 = r0 * r32;
    r8 = fmaf(r2, r8, r40 * r54);
    r9 = r1 * r57;
    r42 = r40 * r51;
    r42 = fmaf(r29, r42, r2 * r9);
    r9 = r0 * r55;
    r9 = fmaf(r2, r9, r38 * r54);
    r7 = r1 * r35;
    r62 = r38 * r51;
    r62 = fmaf(r29, r62, r2 * r7);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r8,
                                          r42,
                                          r9,
                                          r62);
    r7 = r6 * r5;
    r80 = r6 * r4;
    r80 = fmaf(r69, r80, r77 * r7);
    r7 = r6 * r5;
    r74 = r6 * r4;
    r74 = fmaf(r71, r74, r20 * r7);
    r7 = r6 * r4;
    r18 = r6 * r5;
    r18 = fmaf(r75, r18, r39 * r7);
    r7 = r6 * r5;
    r65 = r6 * r4;
    r65 = fmaf(r58, r65, r63 * r7);
    WriteSum4<float, float>((float*)inout_shared, r80, r74, r18, r65);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r6 * r5;
    r18 = r6 * r4;
    r18 = fmaf(r8, r18, r42 * r65);
    r65 = r6 * r4;
    r74 = r6 * r5;
    r74 = fmaf(r62, r74, r9 * r65);
    WriteSum2<float, float>((float*)inout_shared, r18, r74);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fmaf(r77, r77, r69 * r69);
    r18 = fmaf(r71, r71, r20 * r20);
    r65 = fmaf(r75, r75, r39 * r39);
    r80 = fmaf(r63, r63, r58 * r58);
    WriteSum4<float, float>((float*)inout_shared, r74, r18, r65, r80);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fmaf(r8, r8, r42 * r42);
    r65 = fmaf(r62, r62, r9 * r9);
    WriteSum2<float, float>((float*)inout_shared, r80, r65);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fmaf(r69, r71, r77 * r20);
    r80 = fmaf(r69, r39, r77 * r75);
    r18 = fmaf(r77, r63, r69 * r58);
    r74 = fmaf(r77, r42, r69 * r8);
    WriteSum4<float, float>((float*)inout_shared, r65, r80, r18, r74);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fmaf(r77, r62, r69 * r9);
    r69 = fmaf(r20, r75, r71 * r39);
    r74 = fmaf(r71, r58, r20 * r63);
    r18 = fmaf(r20, r42, r71 * r8);
    WriteSum4<float, float>((float*)inout_shared, r77, r69, r74, r18);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = fmaf(r71, r9, r20 * r62);
    r20 = fmaf(r39, r58, r75 * r63);
    r18 = fmaf(r75, r42, r39 * r8);
    r39 = fmaf(r39, r9, r75 * r62);
    WriteSum4<float, float>((float*)inout_shared, r71, r20, r18, r39);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fmaf(r58, r8, r63 * r42);
    r63 = fmaf(r63, r62, r58 * r9);
    r62 = fmaf(r42, r62, r8 * r9);
    WriteSum3<float, float>((float*)inout_shared, r39, r63, r62);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = r44 * r0;
    r62 = r62 * r47;
    r62 = fmaf(r2, r62, r54 * r46);
    r63 = r45 * r1;
    r63 = r63 * r47;
    r47 = r51 * r29;
    r47 = fmaf(r46, r47, r2 * r63);
    WriteIdx2<1024, float, float, float2>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r62,
        r47);
    r63 = r6 * r4;
    r46 = r6 * r5;
    r46 = fmaf(r47, r46, r62 * r63);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_njtr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r46);
  if (global_thread_idx < problem_size) {
    r47 = fmaf(r47, r47, r62 * r62);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r47);
  if (global_thread_idx < problem_size) {
    r47 = r0 * r52;
    r47 = fmaf(r28, r54, r2 * r47);
    r62 = r1 * r3;
    r46 = r28 * r51;
    r46 = fmaf(r29, r46, r2 * r62);
    r62 = r0 * r53;
    r62 = fmaf(r41, r54, r2 * r62);
    r63 = r41 * r51;
    r39 = r1 * r34;
    r39 = fmaf(r2, r39, r29 * r63);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r47,
                                          r46,
                                          r62,
                                          r39);
    r63 = r0 * r22;
    r63 = fmaf(r2, r63, r49 * r54);
    r54 = r49 * r51;
    r42 = r1 * r43;
    r42 = fmaf(r2, r42, r29 * r54);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r63,
                                          r42);
    r54 = r6 * r5;
    r2 = r6 * r4;
    r2 = fmaf(r47, r2, r46 * r54);
    r54 = r6 * r5;
    r9 = r6 * r4;
    r9 = fmaf(r62, r9, r39 * r54);
    r54 = r6 * r5;
    r8 = r6 * r4;
    r8 = fmaf(r63, r8, r42 * r54);
    WriteSum3<float, float>((float*)inout_shared, r2, r9, r8);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r46, r46, r47 * r47);
    r9 = fmaf(r39, r39, r62 * r62);
    r2 = fmaf(r42, r42, r63 * r63);
    WriteSum3<float, float>((float*)inout_shared, r8, r9, r2);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r46, r39, r47 * r62);
    r46 = fmaf(r46, r42, r47 * r63);
    r42 = fmaf(r39, r42, r62 * r63);
    WriteSum3<float, float>((float*)inout_shared, r2, r46, r42);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_njtr_local,
                       out_sensor_from_rig_log_scale_njtr,
                       1);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                       out_sensor_from_rig_log_scale_precond_diag,
                       1);
}

void FixedRigPinholeResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    const float* const sensor_from_rig_log_scale,
    float* calib,
    unsigned int calib_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_sensor_from_rig_log_scale_jac,
    unsigned int out_sensor_from_rig_log_scale_jac_num_alloc,
    float* const out_sensor_from_rig_log_scale_njtr,
    float* const out_sensor_from_rig_log_scale_precond_diag,
    float* const out_sensor_from_rig_log_scale_precond_tril,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
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