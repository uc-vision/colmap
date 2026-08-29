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
      r76, r77, r78, r79, r80;
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
    r18 = r11 * r14;
    r19 = fmaf(r12, r17, r18);
    r20 = r10 * r15;
    r19 = fmaf(r5, r20, r19);
    r21 = r13 * r16;
    r19 = r19 + r21;
    r20 = r9 * r19;
    r22 = fmaf(r13, r14, r10 * r17);
    r23 = r12 * r15;
    r24 = r11 * r16;
    r22 = fmaf(r5, r24, r22);
    r22 = r22 + r23;
    r20 = r20 * r22;
    r24 = fmaf(r11, r15, r10 * r14);
    r24 = fmaf(r12, r16, r24);
    r24 = fmaf(r13, r17, r5 * r24);
    r25 = r12 * r14;
    r25 = fmaf(r5, r25, r11 * r17);
    r25 = fmaf(r13, r15, r25);
    r25 = fmaf(r10, r16, r25);
    r26 = -2.00000000000000000e+00;
    r27 = r25 * r26;
    r28 = fmaf(r24, r27, r20);
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
    r32 = r32 * r26;
    r33 = 1.00000000000000000e+00;
    r34 = r15 * r15;
    r34 = fmaf(r26, r34, r33);
    r35 = r32 + r34;
    r36 = fmaf(r31, r35, r6 * r28);
    r37 = r14 * r16;
    r37 = r37 * r9;
    r38 = r15 * r17;
    r38 = fmaf(r26, r38, r37);
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
    r46 = r9 * r19;
    r46 = r46 * r25;
    r47 = r9 * r24;
    r48 = fmaf(r22, r47, r46);
    r49 = r25 * r27;
    r50 = r33 + r49;
    r51 = r22 * r22;
    r51 = r51 * r26;
    r50 = r50 + r51;
    r36 = fmaf(r29, r38, r36);
    r36 = fmaf(r30, r39, r36);
    r36 = r36 + r43;
    r36 = fmaf(r7, r48, r36);
    r36 = fmaf(r8, r50, r36);
    r52 = copysign(1.0, r36);
    r52 = fmaf(r1, r52, r36);
    r1 = 1.0 / r52;
    r20 = fmaf(r25, r47, r20);
    r36 = r16 * r16;
    r36 = r36 * r26;
    r34 = r36 + r34;
    r53 = fmaf(r29, r34, r8 * r20);
    r54 = r15 * r17;
    r54 = fmaf(r9, r54, r37);
    r37 = r16 * r17;
    r55 = r14 * r15;
    r55 = r55 * r9;
    r37 = fmaf(r26, r37, r55);
    r49 = r33 + r49;
    r56 = r19 * r19;
    r56 = r56 * r26;
    r49 = r49 + r56;
    r57 = r9 * r22;
    r57 = r57 * r25;
    r58 = r19 * r24;
    r58 = fmaf(r26, r58, r57);
    r53 = fmaf(r31, r54, r53);
    r53 = fmaf(r30, r37, r53);
    r53 = fmaf(r41, r45, r53);
    r53 = fmaf(r6, r49, r53);
    r53 = fmaf(r7, r58, r53);
    r53 = r44 * r53;
    r3 = fmaf(r1, r53, r3);
    r4 = fmaf(r4, r5, r2);
    r56 = r33 + r56;
    r56 = r56 + r51;
    r36 = r33 + r36;
    r36 = r36 + r32;
    r30 = fmaf(r30, r36, r7 * r56);
    r32 = r16 * r17;
    r32 = fmaf(r9, r32, r55);
    r55 = r14 * r17;
    r55 = fmaf(r26, r55, r40);
    r40 = r22 * r24;
    r40 = fmaf(r26, r40, r46);
    r57 = fmaf(r19, r47, r57);
    r30 = fmaf(r29, r32, r30);
    r30 = fmaf(r31, r55, r30);
    r30 = fmaf(r42, r45, r30);
    r30 = fmaf(r8, r40, r30);
    r30 = fmaf(r6, r57, r30);
    r30 = r0 * r30;
    r4 = fmaf(r1, r30, r4);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r4);
    r31 = fmaf(r3, r3, r4 * r4);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r31);
  if (global_thread_idx < problem_size) {
    r31 = -4.00000000000000000e+00;
    r29 = r25 * r31;
    r46 = 5.00000000000000000e-01;
    r33 = r17 * r46;
    r51 = fmaf(r46, r18, r12 * r33);
    r2 = -5.00000000000000000e-01;
    r59 = r10 * r2;
    r51 = fmaf(r15, r59, r51);
    r51 = fmaf(r46, r21, r51);
    r29 = r29 * r51;
    r60 = r11 * r17;
    r61 = r12 * r14;
    r61 = fmaf(r46, r61, r2 * r60);
    r60 = r13 * r15;
    r61 = fmaf(r2, r60, r61);
    r61 = fmaf(r16, r59, r61);
    r60 = r19 * r31;
    r62 = r61 * r60;
    r63 = r29 + r62;
    r64 = r9 * r25;
    r65 = r11 * r15;
    r66 = r12 * r16;
    r66 = fmaf(r2, r66, r2 * r65);
    r66 = fmaf(r13, r33, r66);
    r66 = fmaf(r14, r59, r66);
    r64 = r64 * r66;
    r65 = r19 * r26;
    r67 = r13 * r14;
    r68 = r11 * r16;
    r68 = fmaf(r46, r68, r2 * r67);
    r68 = fmaf(r17, r59, r68);
    r68 = fmaf(r2, r23, r68);
    r65 = fmaf(r68, r65, r64);
    r59 = r9 * r22;
    r59 = r59 * r51;
    r67 = r24 * r26;
    r65 = fmaf(r61, r67, r65);
    r65 = r65 + r59;
    r65 = fmaf(r7, r65, r6 * r63);
    r63 = r9 * r25;
    r63 = fmaf(r51, r47, r68 * r63);
    r67 = r9 * r19;
    r67 = r67 * r66;
    r69 = r9 * r22;
    r69 = fmaf(r61, r69, r67);
    r63 = r63 + r69;
    r65 = fmaf(r8, r63, r65);
    r63 = r44 * r65;
    r70 = r24 * r26;
    r70 = fmaf(r68, r27, r51 * r70);
    r70 = r70 + r69;
    r71 = r9 * r25;
    r71 = r71 * r61;
    r72 = r9 * r19;
    r72 = fmaf(r51, r72, r71);
    r51 = r9 * r22;
    r51 = r51 * r68;
    r73 = r66 * r47;
    r74 = r51 + r73;
    r75 = r72 + r74;
    r75 = fmaf(r7, r75, r6 * r70);
    r70 = r22 * r31;
    r70 = r70 * r66;
    r29 = r29 + r70;
    r75 = fmaf(r8, r29, r75);
    r52 = r52 * r52;
    r52 = 1.0 / r52;
    r52 = r5 * r52;
    r53 = r52 * r53;
    r63 = fmaf(r75, r53, r1 * r63);
    r62 = r70 + r62;
    r59 = fmaf(r61, r47, r59);
    r70 = r9 * r19;
    r70 = fmaf(r68, r70, r64);
    r59 = r59 + r70;
    r59 = fmaf(r6, r59, r7 * r62);
    r62 = r22 * r26;
    r64 = r24 * r26;
    r64 = r64 * r66;
    r62 = fmaf(r68, r62, r64);
    r62 = r62 + r72;
    r59 = fmaf(r8, r62, r59);
    r62 = r0 * r59;
    r72 = r75 * r52;
    r72 = fmaf(r30, r72, r1 * r62);
    r62 = r25 * r31;
    r62 = r62 * r66;
    r29 = r13 * r14;
    r76 = r11 * r16;
    r76 = fmaf(r2, r76, r46 * r29);
    r76 = fmaf(r10, r33, r76);
    r76 = fmaf(r46, r23, r76);
    r23 = r76 * r60;
    r29 = r62 + r23;
    r73 = r71 + r73;
    r71 = r9 * r19;
    r77 = r12 * r17;
    r78 = r10 * r15;
    r78 = fmaf(r46, r78, r2 * r77);
    r78 = fmaf(r2, r18, r78);
    r78 = fmaf(r2, r21, r78);
    r71 = r71 * r78;
    r21 = r9 * r22;
    r21 = fmaf(r76, r21, r71);
    r73 = r73 + r21;
    r73 = fmaf(r8, r73, r6 * r29);
    r29 = r19 * r26;
    r18 = r24 * r26;
    r18 = fmaf(r76, r18, r61 * r29);
    r29 = r9 * r25;
    r77 = r9 * r22;
    r77 = r77 * r66;
    r29 = fmaf(r78, r29, r77);
    r18 = r18 + r29;
    r73 = fmaf(r7, r18, r73);
    r18 = r44 * r73;
    r79 = r9 * r25;
    r79 = r79 * r76;
    r80 = fmaf(r78, r47, r79);
    r80 = r80 + r69;
    r69 = fmaf(r61, r27, r64);
    r69 = r69 + r21;
    r69 = fmaf(r6, r69, r7 * r80);
    r80 = r22 * r78;
    r21 = r31 * r80;
    r62 = r62 + r21;
    r69 = fmaf(r8, r62, r69);
    r18 = fmaf(r69, r53, r1 * r18);
    r21 = r23 + r21;
    r79 = r67 + r79;
    r67 = r22 * r26;
    r79 = fmaf(r61, r67, r79);
    r23 = r24 * r26;
    r79 = fmaf(r78, r23, r79);
    r79 = fmaf(r8, r79, r7 * r21);
    r21 = r9 * r19;
    r76 = fmaf(r76, r47, r61 * r21);
    r76 = r76 + r29;
    r79 = fmaf(r6, r76, r79);
    r76 = r0 * r79;
    r21 = r69 * r52;
    r21 = fmaf(r30, r21, r1 * r76);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r63,
                                          r72,
                                          r18,
                                          r21);
    r76 = r12 * r14;
    r61 = r13 * r15;
    r61 = fmaf(r46, r61, r2 * r76);
    r76 = r10 * r16;
    r61 = fmaf(r46, r76, r61);
    r61 = fmaf(r11, r33, r61);
    r33 = fmaf(r61, r47, r9 * r80);
    r33 = r33 + r70;
    r76 = r9 * r19;
    r76 = r76 * r61;
    r77 = r77 + r76;
    r46 = r24 * r26;
    r77 = fmaf(r68, r46, r77);
    r77 = fmaf(r78, r27, r77);
    r77 = fmaf(r6, r77, r7 * r33);
    r33 = r25 * r31;
    r33 = r33 * r68;
    r31 = r22 * r31;
    r31 = r31 * r61;
    r27 = r33 + r31;
    r77 = fmaf(r8, r27, r77);
    r60 = r66 * r60;
    r33 = r33 + r60;
    r64 = r51 + r64;
    r51 = r9 * r25;
    r51 = r51 * r61;
    r66 = r19 * r26;
    r64 = fmaf(r78, r66, r64);
    r64 = r64 + r51;
    r64 = fmaf(r7, r64, r6 * r33);
    r47 = fmaf(r68, r47, r76);
    r47 = r47 + r29;
    r64 = fmaf(r8, r47, r64);
    r47 = r44 * r64;
    r47 = fmaf(r1, r47, r77 * r53);
    r29 = r77 * r52;
    r60 = r31 + r60;
    r31 = r24 * r26;
    r80 = fmaf(r26, r80, r61 * r31);
    r80 = r80 + r70;
    r80 = fmaf(r8, r80, r7 * r60);
    r51 = r71 + r51;
    r51 = r51 + r74;
    r80 = fmaf(r6, r51, r80);
    r51 = r0 * r80;
    r51 = fmaf(r1, r51, r30 * r29);
    r29 = r44 * r34;
    r29 = fmaf(r38, r53, r1 * r29);
    r6 = r38 * r52;
    r74 = r0 * r32;
    r74 = fmaf(r1, r74, r30 * r6);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r47,
                                          r51,
                                          r29,
                                          r74);
    r6 = r44 * r37;
    r6 = fmaf(r1, r6, r39 * r53);
    r71 = r0 * r36;
    r8 = r39 * r52;
    r8 = fmaf(r30, r8, r1 * r71);
    r71 = r44 * r54;
    r71 = fmaf(r35, r53, r1 * r71);
    r60 = r35 * r52;
    r7 = r0 * r55;
    r7 = fmaf(r1, r7, r30 * r60);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r6,
                                          r8,
                                          r71,
                                          r7);
    r60 = r5 * r4;
    r70 = r5 * r3;
    r70 = fmaf(r63, r70, r72 * r60);
    r60 = r5 * r4;
    r31 = r5 * r3;
    r31 = fmaf(r18, r31, r21 * r60);
    r60 = r5 * r3;
    r61 = r5 * r4;
    r61 = fmaf(r51, r61, r47 * r60);
    r60 = r5 * r3;
    r68 = r5 * r4;
    r68 = fmaf(r74, r68, r29 * r60);
    WriteSum4<float, float>((float*)inout_shared, r70, r31, r61, r68);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r5 * r4;
    r61 = r5 * r3;
    r61 = fmaf(r6, r61, r8 * r68);
    r68 = r5 * r4;
    r31 = r5 * r3;
    r31 = fmaf(r71, r31, r7 * r68);
    WriteSum2<float, float>((float*)inout_shared, r61, r31);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r63, r63, r72 * r72);
    r61 = fmaf(r21, r21, r18 * r18);
    r68 = fmaf(r51, r51, r47 * r47);
    r70 = fmaf(r74, r74, r29 * r29);
    WriteSum4<float, float>((float*)inout_shared, r31, r61, r68, r70);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fmaf(r8, r8, r6 * r6);
    r68 = fmaf(r71, r71, r7 * r7);
    WriteSum2<float, float>((float*)inout_shared, r70, r68);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fmaf(r63, r18, r72 * r21);
    r70 = fmaf(r63, r47, r72 * r51);
    r61 = fmaf(r72, r74, r63 * r29);
    r31 = fmaf(r63, r6, r72 * r8);
    WriteSum4<float, float>((float*)inout_shared, r68, r70, r61, r31);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fmaf(r72, r7, r63 * r71);
    r63 = fmaf(r21, r51, r18 * r47);
    r31 = fmaf(r18, r29, r21 * r74);
    r61 = fmaf(r18, r6, r21 * r8);
    WriteSum4<float, float>((float*)inout_shared, r72, r63, r31, r61);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = fmaf(r18, r71, r21 * r7);
    r21 = fmaf(r47, r29, r51 * r74);
    r61 = fmaf(r51, r8, r47 * r6);
    r51 = fmaf(r51, r7, r47 * r71);
    WriteSum4<float, float>((float*)inout_shared, r18, r21, r61, r51);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fmaf(r74, r8, r29 * r6);
    r74 = fmaf(r74, r7, r29 * r71);
    r7 = fmaf(r8, r7, r6 * r71);
    WriteSum3<float, float>((float*)inout_shared, r51, r74, r7);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r41 * r44;
    r7 = r7 * r45;
    r7 = fmaf(r53, r43, r1 * r7);
    r74 = r42 * r0;
    r74 = r74 * r45;
    r45 = r52 * r30;
    r45 = fmaf(r43, r45, r1 * r74);
    WriteIdx2<1024, float, float, float2>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r7,
        r45);
    r74 = r5 * r3;
    r43 = r5 * r4;
    r43 = fmaf(r45, r43, r7 * r74);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_njtr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r43);
  if (global_thread_idx < problem_size) {
    r7 = fmaf(r7, r7, r45 * r45);
  };
  SumStore<float>(out_sensor_from_rig_log_scale_precond_diag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r7);
  if (global_thread_idx < problem_size) {
    r7 = r44 * r49;
    r7 = fmaf(r28, r53, r1 * r7);
    r45 = r28 * r52;
    r43 = r0 * r57;
    r43 = fmaf(r1, r43, r30 * r45);
    r45 = r44 * r58;
    r45 = fmaf(r1, r45, r48 * r53);
    r74 = r0 * r56;
    r51 = r48 * r52;
    r51 = fmaf(r30, r51, r1 * r74);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r7,
                                          r43,
                                          r45,
                                          r51);
    r74 = r44 * r20;
    r74 = fmaf(r1, r74, r50 * r53);
    r53 = r50 * r52;
    r8 = r0 * r40;
    r8 = fmaf(r1, r8, r30 * r53);
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r74, r8);
    r53 = r5 * r4;
    r1 = r5 * r3;
    r1 = fmaf(r7, r1, r43 * r53);
    r53 = r5 * r3;
    r71 = r5 * r4;
    r71 = fmaf(r51, r71, r45 * r53);
    r53 = r5 * r3;
    r6 = r5 * r4;
    r6 = fmaf(r8, r6, r74 * r53);
    WriteSum3<float, float>((float*)inout_shared, r1, r71, r6);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fmaf(r7, r7, r43 * r43);
    r71 = fmaf(r51, r51, r45 * r45);
    r1 = fmaf(r74, r74, r8 * r8);
    WriteSum3<float, float>((float*)inout_shared, r6, r71, r1);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r7, r45, r43 * r51);
    r7 = fmaf(r7, r74, r43 * r8);
    r8 = fmaf(r51, r8, r45 * r74);
    WriteSum3<float, float>((float*)inout_shared, r1, r7, r8);
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