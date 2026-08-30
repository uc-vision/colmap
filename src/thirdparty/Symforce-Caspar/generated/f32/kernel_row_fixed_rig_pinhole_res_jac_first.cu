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
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_calibration,
    unsigned int sensor_calibration_num_alloc,
    SharedIndex* sensor_calibration_indices,
    const float* const sensor_from_rig_log_scale,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    const float* const reprojection_loss_scale,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
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

  __shared__ float out_rTr_local[1];

  __shared__ float out_sensor_from_rig_log_scale_njtr_local[1];

  __shared__ float out_sensor_from_rig_log_scale_precond_diag_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85;
  LoadShared<3, float, float>(sensor_calibration,
                              8 * sensor_calibration_num_alloc,
                              sensor_calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       sensor_calibration_indices_loc[threadIdx.x].target,
                       r0,
                       r1,
                       r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r4);
    r5 = -1.00000000000000000e+00;
    r3 = fmaf(r3, r5, r1);
    r1 = 9.99999999999999955e-07;
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r6,
                       r7,
                       r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = 2.00000000000000000e+00;
  };
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
  LoadShared<4, float, float>(sensor_calibration,
                              0 * sensor_calibration_num_alloc,
                              sensor_calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       sensor_calibration_indices_loc[threadIdx.x].target,
                       r14,
                       r15,
                       r16,
                       r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r18 = r12 * r17;
    r19 = r11 * r14;
    r20 = r18 + r19;
    r21 = r10 * r15;
    r20 = fmaf(r13, r16, r20);
    r20 = fmaf(r5, r21, r20);
    r22 = r9 * r20;
    r23 = fmaf(r13, r14, r10 * r17);
    r24 = r11 * r16;
    r23 = fmaf(r5, r24, r23);
    r23 = fmaf(r12, r15, r23);
    r22 = r22 * r23;
    r24 = r12 * r14;
    r24 = fmaf(r5, r24, r11 * r17);
    r24 = fmaf(r13, r15, r24);
    r24 = fmaf(r10, r16, r24);
    r25 = -2.00000000000000000e+00;
    r26 = fmaf(r11, r15, r10 * r14);
    r26 = fmaf(r12, r16, r26);
    r26 = fmaf(r13, r17, r5 * r26);
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
    r32 = r14 * r14;
    r32 = r32 * r25;
    r33 = 1.00000000000000000e+00;
    r34 = r15 * r15;
    r34 = fmaf(r25, r34, r33);
    r35 = r32 + r34;
    r36 = fmaf(r31, r35, r6 * r28);
    r37 = r15 * r17;
    r38 = r14 * r16;
    r38 = r38 * r9;
    r37 = fmaf(r25, r37, r38);
    r39 = r14 * r17;
    r40 = r15 * r16;
    r40 = r40 * r9;
    r39 = fmaf(r9, r39, r40);
  };
  LoadShared<4, float, float>(sensor_calibration,
                              4 * sensor_calibration_num_alloc,
                              sensor_calibration_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       sensor_calibration_indices_loc[threadIdx.x].target,
                       r41,
                       r42,
                       r43,
                       r44);
  };
  __syncthreads();
  LoadUnique<1, float, float>(
      sensor_from_rig_log_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r45);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r46 = 2.71828182845904523536;
    r45 = powf(r46, r45);
    r43 = r43 * r45;
    r46 = r9 * r20;
    r46 = r46 * r24;
    r47 = r9 * r23;
    r47 = fmaf(r26, r47, r46);
    r48 = r25 * r24;
    r48 = r48 * r24;
    r49 = r33 + r48;
    r50 = r25 * r23;
    r50 = r50 * r23;
    r49 = r49 + r50;
    r36 = fmaf(r29, r37, r36);
    r36 = fmaf(r30, r39, r36);
    r36 = r36 + r43;
    r36 = fmaf(r7, r47, r36);
    r36 = fmaf(r8, r49, r36);
    r51 = copysign(1.0, r36);
    r51 = fmaf(r1, r51, r36);
    r1 = 1.0 / r51;
    r36 = r9 * r24;
    r36 = fmaf(r26, r36, r22);
    r22 = r16 * r16;
    r22 = r25 * r22;
    r34 = r22 + r34;
    r52 = fmaf(r29, r34, r8 * r36);
    r53 = r15 * r17;
    r53 = fmaf(r9, r53, r38);
    r38 = r14 * r15;
    r38 = r38 * r9;
    r54 = r16 * r17;
    r54 = fmaf(r25, r54, r38);
    r48 = r33 + r48;
    r55 = r25 * r20;
    r55 = r55 * r20;
    r48 = r48 + r55;
    r56 = r9 * r23;
    r56 = r56 * r24;
    r57 = fmaf(r20, r27, r56);
    r52 = fmaf(r31, r53, r52);
    r52 = fmaf(r30, r54, r52);
    r52 = fmaf(r41, r45, r52);
    r52 = fmaf(r6, r48, r52);
    r52 = fmaf(r7, r57, r52);
    r52 = r44 * r52;
    r3 = fmaf(r1, r52, r3);
  };
  LoadUnique<1, float, float>(reprojection_loss_scale, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r58);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r58 = r58 * r58;
    r58 = 1.0 / r58;
    r4 = fmaf(r4, r5, r2);
    r55 = r33 + r55;
    r55 = r55 + r50;
    r22 = r33 + r22;
    r22 = r22 + r32;
    r30 = fmaf(r30, r22, r7 * r55);
    r32 = r16 * r17;
    r32 = fmaf(r9, r32, r38);
    r38 = r14 * r17;
    r38 = fmaf(r25, r38, r40);
    r46 = fmaf(r23, r27, r46);
    r40 = r9 * r20;
    r40 = fmaf(r26, r40, r56);
    r30 = fmaf(r29, r32, r30);
    r30 = fmaf(r31, r38, r30);
    r30 = fmaf(r42, r45, r30);
    r30 = fmaf(r8, r46, r30);
    r30 = fmaf(r6, r40, r30);
    r30 = r0 * r30;
    r4 = fmaf(r1, r30, r4);
    r31 = fmaf(r3, r3, r4 * r4);
    r31 = fmaf(r31, r58, r33);
    r29 = sqrtf(r31);
    r29 = r33 + r29;
    r33 = 1.0 / r29;
    r56 = r9 * r33;
    r50 = sqrtf(r56);
    r2 = r3 * r50;
    r59 = r4 * r50;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r59);
    r59 = r9 * r3;
    r59 = r59 * r3;
    r60 = r4 * r33;
    r61 = r9 * r4;
    r60 = fmaf(r61, r60, r33 * r59);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r60);
  if (global_thread_idx < problem_size) {
    r60 = r9 * r3;
    r59 = -4.00000000000000000e+00;
    r62 = r24 * r59;
    r63 = 5.00000000000000000e-01;
    r64 = fmaf(r63, r19, r63 * r18);
    r65 = -5.00000000000000000e-01;
    r66 = r13 * r63;
    r64 = fmaf(r65, r21, r64);
    r64 = fmaf(r16, r66, r64);
    r62 = r62 * r64;
    r67 = r11 * r17;
    r68 = r12 * r14;
    r68 = fmaf(r63, r68, r65 * r67);
    r67 = r13 * r15;
    r68 = fmaf(r65, r67, r68);
    r69 = r10 * r16;
    r68 = fmaf(r65, r69, r68);
    r69 = r20 * r59;
    r67 = r68 * r69;
    r70 = r62 + r67;
    r71 = r9 * r24;
    r72 = r10 * r14;
    r73 = r11 * r15;
    r73 = fmaf(r65, r73, r65 * r72);
    r72 = r12 * r16;
    r73 = fmaf(r65, r72, r73);
    r73 = fmaf(r17, r66, r73);
    r71 = r71 * r73;
    r72 = r9 * r23;
    r72 = r72 * r64;
    r74 = r71 + r72;
    r75 = r25 * r20;
    r76 = r10 * r17;
    r77 = r13 * r14;
    r77 = fmaf(r65, r77, r65 * r76);
    r76 = r12 * r15;
    r77 = fmaf(r65, r76, r77);
    r78 = r11 * r16;
    r77 = fmaf(r63, r78, r77);
    r74 = fmaf(r77, r75, r74);
    r74 = fmaf(r68, r27, r74);
    r74 = fmaf(r7, r74, r6 * r70);
    r70 = r9 * r26;
    r75 = r24 * r77;
    r70 = fmaf(r9, r75, r64 * r70);
    r78 = r9 * r20;
    r78 = r78 * r73;
    r76 = r9 * r23;
    r76 = fmaf(r68, r76, r78);
    r70 = r70 + r76;
    r74 = fmaf(r8, r70, r74);
    r70 = r44 * r74;
    r79 = fmaf(r64, r27, r25 * r75);
    r79 = r79 + r76;
    r80 = r9 * r24;
    r80 = r80 * r68;
    r81 = r9 * r20;
    r81 = fmaf(r64, r81, r80);
    r64 = r9 * r23;
    r64 = r64 * r77;
    r82 = r9 * r26;
    r82 = r82 * r73;
    r83 = r64 + r82;
    r84 = r81 + r83;
    r84 = fmaf(r7, r84, r6 * r79);
    r79 = r23 * r59;
    r79 = r79 * r73;
    r62 = r79 + r62;
    r84 = fmaf(r8, r62, r84);
    r51 = r51 * r51;
    r51 = 1.0 / r51;
    r51 = r5 * r51;
    r52 = r51 * r52;
    r70 = fmaf(r84, r52, r1 * r70);
    r67 = r79 + r67;
    r79 = r9 * r26;
    r79 = fmaf(r68, r79, r72);
    r72 = r9 * r20;
    r72 = fmaf(r77, r72, r71);
    r79 = r79 + r72;
    r79 = fmaf(r6, r79, r7 * r67);
    r67 = r25 * r23;
    r71 = r73 * r27;
    r67 = fmaf(r77, r67, r71);
    r67 = r67 + r81;
    r79 = fmaf(r8, r67, r79);
    r67 = r0 * r79;
    r81 = r84 * r51;
    r81 = fmaf(r30, r81, r1 * r67);
    r60 = fmaf(r81, r61, r70 * r60);
    r58 = r65 * r58;
    r56 = rsqrtf(r56);
    r29 = r29 * r29;
    r29 = 1.0 / r29;
    r31 = rsqrtf(r31);
    r58 = r58 * r56;
    r58 = r58 * r29;
    r58 = r58 * r31;
    r60 = r60 * r58;
    r70 = fmaf(r70, r50, r3 * r60);
    r60 = fmaf(r4, r60, r81 * r50);
    r81 = r24 * r59;
    r81 = r81 * r73;
    r31 = r10 * r17;
    r29 = r12 * r15;
    r29 = fmaf(r63, r29, r63 * r31);
    r31 = r11 * r16;
    r29 = fmaf(r65, r31, r29);
    r29 = fmaf(r14, r66, r29);
    r31 = r29 * r69;
    r56 = r81 + r31;
    r82 = r80 + r82;
    r80 = r9 * r20;
    r67 = r13 * r16;
    r18 = fmaf(r65, r18, r65 * r67);
    r18 = fmaf(r65, r19, r18);
    r18 = fmaf(r63, r21, r18);
    r80 = r80 * r18;
    r21 = r9 * r23;
    r21 = fmaf(r29, r21, r80);
    r82 = r82 + r21;
    r82 = fmaf(r8, r82, r6 * r56);
    r56 = r25 * r20;
    r56 = fmaf(r29, r27, r68 * r56);
    r19 = r9 * r24;
    r67 = r9 * r23;
    r67 = r67 * r73;
    r19 = fmaf(r18, r19, r67);
    r56 = r56 + r19;
    r82 = fmaf(r7, r56, r82);
    r56 = r44 * r82;
    r62 = r9 * r24;
    r62 = r62 * r29;
    r85 = r9 * r26;
    r85 = fmaf(r18, r85, r62);
    r85 = r85 + r76;
    r76 = r25 * r24;
    r76 = fmaf(r68, r76, r71);
    r76 = r76 + r21;
    r76 = fmaf(r6, r76, r7 * r85);
    r85 = r23 * r18;
    r21 = r59 * r85;
    r81 = r81 + r21;
    r76 = fmaf(r8, r81, r76);
    r56 = fmaf(r76, r52, r1 * r56);
    r81 = r9 * r3;
    r21 = r31 + r21;
    r62 = r78 + r62;
    r78 = r25 * r23;
    r62 = fmaf(r68, r78, r62);
    r62 = fmaf(r18, r27, r62);
    r62 = fmaf(r8, r62, r7 * r21);
    r21 = r9 * r20;
    r78 = r9 * r26;
    r78 = fmaf(r29, r78, r68 * r21);
    r78 = r78 + r19;
    r62 = fmaf(r6, r78, r62);
    r78 = r0 * r62;
    r21 = r76 * r51;
    r21 = fmaf(r30, r21, r1 * r78);
    r81 = fmaf(r21, r61, r56 * r81);
    r78 = r3 * r81;
    r78 = fmaf(r58, r78, r56 * r50);
    r56 = r4 * r81;
    r56 = fmaf(r58, r56, r21 * r50);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r70,
                                          r60,
                                          r78,
                                          r56);
    r21 = r9 * r26;
    r29 = r11 * r17;
    r68 = r12 * r14;
    r68 = fmaf(r65, r68, r63 * r29);
    r29 = r10 * r16;
    r68 = fmaf(r63, r29, r68);
    r68 = fmaf(r15, r66, r68);
    r21 = fmaf(r9, r85, r68 * r21);
    r21 = r21 + r72;
    r66 = r25 * r24;
    r66 = fmaf(r18, r66, r67);
    r67 = r9 * r20;
    r67 = r67 * r68;
    r66 = r66 + r67;
    r66 = fmaf(r77, r27, r66);
    r66 = fmaf(r6, r66, r7 * r21);
    r21 = r23 * r59;
    r21 = r21 * r68;
    r75 = r59 * r75;
    r59 = r21 + r75;
    r66 = fmaf(r8, r59, r66);
    r69 = r73 * r69;
    r75 = r75 + r69;
    r73 = r9 * r24;
    r73 = r73 * r68;
    r64 = r64 + r73;
    r59 = r25 * r20;
    r64 = fmaf(r18, r59, r64);
    r64 = r64 + r71;
    r64 = fmaf(r7, r64, r6 * r75);
    r75 = r9 * r26;
    r75 = fmaf(r77, r75, r67);
    r75 = r75 + r19;
    r64 = fmaf(r8, r75, r64);
    r75 = r44 * r64;
    r75 = fmaf(r1, r75, r66 * r52);
    r19 = r9 * r3;
    r67 = r66 * r51;
    r69 = r21 + r69;
    r27 = fmaf(r68, r27, r25 * r85);
    r27 = r27 + r72;
    r27 = fmaf(r8, r27, r7 * r69);
    r73 = r80 + r73;
    r73 = r73 + r83;
    r27 = fmaf(r6, r73, r27);
    r73 = r0 * r27;
    r73 = fmaf(r1, r73, r30 * r67);
    r19 = fmaf(r73, r61, r75 * r19);
    r67 = r3 * r19;
    r67 = fmaf(r58, r67, r75 * r50);
    r75 = r4 * r19;
    r73 = fmaf(r73, r50, r58 * r75);
    r75 = r9 * r3;
    r6 = r44 * r34;
    r6 = fmaf(r37, r52, r1 * r6);
    r83 = r37 * r51;
    r80 = r0 * r32;
    r80 = fmaf(r1, r80, r30 * r83);
    r75 = fmaf(r80, r61, r6 * r75);
    r83 = r3 * r75;
    r6 = fmaf(r6, r50, r58 * r83);
    r83 = r4 * r75;
    r83 = fmaf(r58, r83, r80 * r50);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r67,
                                          r73,
                                          r6,
                                          r83);
    r80 = r9 * r3;
    r8 = r44 * r54;
    r8 = fmaf(r1, r8, r39 * r52);
    r69 = r0 * r22;
    r7 = r39 * r51;
    r7 = fmaf(r30, r7, r1 * r69);
    r80 = fmaf(r7, r61, r8 * r80);
    r69 = r3 * r80;
    r8 = fmaf(r8, r50, r58 * r69);
    r69 = r4 * r80;
    r7 = fmaf(r7, r50, r58 * r69);
    r69 = r9 * r3;
    r72 = r44 * r53;
    r72 = fmaf(r35, r52, r1 * r72);
    r68 = r35 * r51;
    r85 = r0 * r38;
    r85 = fmaf(r1, r85, r30 * r68);
    r69 = fmaf(r85, r61, r72 * r69);
    r68 = r3 * r69;
    r72 = fmaf(r72, r50, r58 * r68);
    r68 = r4 * r69;
    r85 = fmaf(r85, r50, r58 * r68);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r8,
                                          r7,
                                          r72,
                                          r85);
    r68 = r5 * r4;
    r68 = r68 * r60;
    r2 = r5 * r2;
    r68 = fmaf(r70, r2, r50 * r68);
    r21 = r5 * r4;
    r21 = r21 * r56;
    r21 = fmaf(r78, r2, r50 * r21);
    r77 = r5 * r4;
    r77 = r77 * r73;
    r77 = fmaf(r67, r2, r50 * r77);
    r71 = r5 * r4;
    r71 = r71 * r83;
    r71 = fmaf(r6, r2, r50 * r71);
    WriteSum4<float, float>((float*)inout_shared, r68, r21, r77, r71);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r5 * r4;
    r71 = r71 * r7;
    r71 = fmaf(r50, r71, r8 * r2);
    r77 = r5 * r4;
    r77 = r77 * r85;
    r77 = fmaf(r50, r77, r72 * r2);
    WriteSum2<float, float>((float*)inout_shared, r71, r77);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fmaf(r60, r60, r70 * r70);
    r71 = fmaf(r56, r56, r78 * r78);
    r21 = fmaf(r73, r73, r67 * r67);
    r68 = fmaf(r83, r83, r6 * r6);
    WriteSum4<float, float>((float*)inout_shared, r77, r71, r21, r68);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fmaf(r8, r8, r7 * r7);
    r21 = fmaf(r72, r72, r85 * r85);
    WriteSum2<float, float>((float*)inout_shared, r68, r21);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fmaf(r60, r56, r70 * r78);
    r68 = fmaf(r70, r67, r60 * r73);
    r71 = fmaf(r70, r6, r60 * r83);
    r77 = fmaf(r60, r7, r70 * r8);
    WriteSum4<float, float>((float*)inout_shared, r21, r68, r71, r77);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fmaf(r60, r85, r70 * r72);
    r70 = fmaf(r78, r67, r56 * r73);
    r77 = fmaf(r56, r83, r78 * r6);
    r71 = fmaf(r78, r8, r56 * r7);
    WriteSum4<float, float>((float*)inout_shared, r60, r70, r77, r71);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fmaf(r56, r85, r78 * r72);
    r78 = fmaf(r67, r6, r73 * r83);
    r71 = fmaf(r73, r7, r67 * r8);
    r67 = fmaf(r67, r72, r73 * r85);
    WriteSum4<float, float>((float*)inout_shared, r56, r78, r71, r67);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = fmaf(r83, r7, r6 * r8);
    r83 = fmaf(r83, r85, r6 * r72);
    r72 = fmaf(r8, r72, r7 * r85);
    WriteSum3<float, float>((float*)inout_shared, r67, r83, r72);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r9 * r3;
    r83 = r41 * r44;
    r83 = r83 * r45;
    r83 = fmaf(r52, r43, r1 * r83);
    r67 = r42 * r0;
    r67 = r67 * r45;
    r45 = r51 * r30;
    r45 = fmaf(r43, r45, r1 * r67);
    r72 = fmaf(r45, r61, r83 * r72);
    r67 = r3 * r72;
    r83 = fmaf(r83, r50, r58 * r67);
    r67 = r4 * r72;
    r67 = fmaf(r58, r67, r45 * r50);
    WriteIdx2<1024, float, float, float2>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r83,
        r67);
    r45 = r5 * r4;
    r45 = r45 * r67;
    r45 = fmaf(r50, r45, r83 * r2);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_njtr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r45);
  if (global_thread_idx < problem_size) {
    r83 = fmaf(r83, r83, r67 * r67);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r83);
  if (global_thread_idx < problem_size) {
    r83 = r9 * r3;
    r67 = r44 * r48;
    r67 = fmaf(r28, r52, r1 * r67);
    r45 = r28 * r51;
    r43 = r0 * r40;
    r43 = fmaf(r1, r43, r30 * r45);
    r83 = fmaf(r43, r61, r67 * r83);
    r45 = r3 * r83;
    r67 = fmaf(r67, r50, r58 * r45);
    r45 = r4 * r83;
    r45 = fmaf(r58, r45, r43 * r50);
    r43 = r44 * r57;
    r43 = fmaf(r1, r43, r47 * r52);
    r8 = r9 * r3;
    r85 = r0 * r55;
    r7 = r47 * r51;
    r7 = fmaf(r30, r7, r1 * r85);
    r8 = fmaf(r7, r61, r43 * r8);
    r85 = r3 * r8;
    r85 = fmaf(r58, r85, r43 * r50);
    r43 = r4 * r8;
    r43 = fmaf(r58, r43, r7 * r50);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r67,
                                          r45,
                                          r85,
                                          r43);
    r7 = r9 * r3;
    r6 = r44 * r36;
    r6 = fmaf(r1, r6, r49 * r52);
    r52 = r49 * r51;
    r71 = r0 * r46;
    r71 = fmaf(r1, r71, r30 * r52);
    r61 = fmaf(r71, r61, r6 * r7);
    r7 = r3 * r61;
    r6 = fmaf(r6, r50, r58 * r7);
    r7 = r4 * r61;
    r7 = fmaf(r58, r7, r71 * r50);
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r6, r7);
    r71 = r5 * r4;
    r71 = r71 * r45;
    r71 = fmaf(r50, r71, r67 * r2);
    r58 = r5 * r4;
    r58 = r58 * r43;
    r58 = fmaf(r50, r58, r85 * r2);
    r52 = r5 * r4;
    r52 = r52 * r7;
    r52 = fmaf(r50, r52, r6 * r2);
    WriteSum3<float, float>((float*)inout_shared, r71, r58, r52);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fmaf(r45, r45, r67 * r67);
    r58 = fmaf(r85, r85, r43 * r43);
    r71 = fmaf(r6, r6, r7 * r7);
    WriteSum3<float, float>((float*)inout_shared, r52, r58, r71);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = fmaf(r67, r85, r45 * r43);
    r45 = fmaf(r45, r7, r67 * r6);
    r6 = fmaf(r85, r6, r43 * r7);
    WriteSum3<float, float>((float*)inout_shared, r71, r45, r6);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_njtr_local,
                       out_sensor_from_rig_log_scale_njtr,
                       1);
  SumFlushFinal<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                       out_sensor_from_rig_log_scale_precond_diag,
                       1);
}

void RowFixedRigPinholeResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_calibration,
    unsigned int sensor_calibration_num_alloc,
    SharedIndex* sensor_calibration_indices,
    const float* const sensor_from_rig_log_scale,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    const float* const reprojection_loss_scale,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
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