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
      r76, r77, r78, r79;
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
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r12, r13);
  };
  __syncthreads();
  LoadShared<2, double, double>(sensor_calibration,
                                0 * sensor_calibration_num_alloc,
                                sensor_calibration_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        sensor_calibration_indices_loc[threadIdx.x].target,
                        r14,
                        r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r13 * r14;
    r17 = fma(r8, r11, r16);
    r18 = r12 * r15;
    r17 = fma(r4, r18, r17);
    r19 = r9 * r10;
    r17 = r17 + r19;
    r18 = r7 * r17;
    r20 = fma(r9, r14, r12 * r11);
    r21 = r8 * r15;
    r22 = r13 * r10;
    r20 = fma(r4, r22, r20);
    r20 = r20 + r21;
    r18 = r18 * r20;
    r22 = fma(r13, r15, r12 * r14);
    r22 = fma(r8, r10, r22);
    r22 = fma(r9, r11, r4 * r22);
    r23 = r8 * r14;
    r23 = fma(r4, r23, r13 * r11);
    r23 = fma(r9, r15, r23);
    r23 = fma(r12, r10, r23);
    r24 = -2.00000000000000000e+00;
    r25 = r23 * r24;
    r26 = fma(r22, r25, r18);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r28 = r14 * r14;
    r28 = r28 * r24;
    r29 = 1.00000000000000000e+00;
    r30 = r15 * r15;
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
    r35 = r14 * r10;
    r35 = r35 * r7;
    r36 = r15 * r11;
    r36 = fma(r24, r36, r35);
    r37 = r14 * r11;
    r38 = r15 * r10;
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
    r42 = r7 * r17;
    r42 = r42 * r23;
    r43 = r7 * r22;
    r44 = fma(r20, r43, r42);
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
    r48 = r20 * r20;
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
    r18 = fma(r23, r43, r18);
    r32 = r10 * r10;
    r32 = r32 * r24;
    r30 = r32 + r30;
    r50 = fma(r33, r30, r45 * r18);
    r51 = r15 * r11;
    r51 = fma(r7, r51, r35);
    r35 = r10 * r11;
    r52 = r14 * r15;
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
    r55 = r17 * r17;
    r55 = r55 * r24;
    r46 = r46 + r55;
    r56 = r7 * r20;
    r56 = r56 * r23;
    r57 = r17 * r22;
    r57 = fma(r24, r57, r56);
    r50 = fma(r27, r51, r50);
    r50 = fma(r34, r35, r50);
    r50 = fma(r53, r41, r50);
    r50 = fma(r5, r46, r50);
    r50 = fma(r6, r57, r50);
    r50 = r40 * r50;
    r2 = fma(r1, r50, r2);
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
    r55 = r29 + r55;
    r55 = r55 + r48;
    r32 = r29 + r32;
    r32 = r32 + r28;
    r34 = fma(r34, r32, r6 * r55);
    r28 = r10 * r11;
    r28 = fma(r7, r28, r52);
    r52 = r14 * r11;
    r52 = fma(r24, r52, r38);
    r38 = r20 * r22;
    r38 = fma(r24, r38, r42);
    r56 = fma(r17, r43, r56);
    r34 = fma(r33, r28, r34);
    r34 = fma(r27, r52, r34);
    r34 = fma(r54, r41, r34);
    r34 = fma(r45, r38, r34);
    r34 = fma(r5, r56, r34);
    r34 = r0 * r34;
    r3 = fma(r1, r34, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r27 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r27);
  if (global_thread_idx < problem_size) {
    r27 = -4.00000000000000000e+00;
    r33 = r23 * r27;
    r42 = 5.00000000000000000e-01;
    r29 = r11 * r42;
    r48 = fma(r42, r16, r8 * r29);
    r58 = -5.00000000000000000e-01;
    r59 = r12 * r58;
    r48 = fma(r15, r59, r48);
    r48 = fma(r42, r19, r48);
    r33 = r33 * r48;
    r60 = r13 * r11;
    r61 = r8 * r14;
    r61 = fma(r42, r61, r58 * r60);
    r60 = r9 * r15;
    r61 = fma(r58, r60, r61);
    r61 = fma(r10, r59, r61);
    r60 = r17 * r27;
    r62 = r61 * r60;
    r63 = r33 + r62;
    r64 = r7 * r23;
    r65 = r13 * r15;
    r66 = r8 * r10;
    r66 = fma(r58, r66, r58 * r65);
    r66 = fma(r9, r29, r66);
    r66 = fma(r14, r59, r66);
    r64 = r64 * r66;
    r65 = r17 * r24;
    r67 = r9 * r14;
    r68 = r13 * r10;
    r68 = fma(r42, r68, r58 * r67);
    r68 = fma(r11, r59, r68);
    r68 = fma(r58, r21, r68);
    r65 = fma(r68, r65, r64);
    r59 = r7 * r20;
    r59 = r59 * r48;
    r67 = r22 * r24;
    r65 = fma(r61, r67, r65);
    r65 = r65 + r59;
    r65 = fma(r6, r65, r5 * r63);
    r63 = r7 * r23;
    r63 = fma(r48, r43, r68 * r63);
    r67 = r7 * r17;
    r67 = r67 * r66;
    r69 = r7 * r20;
    r69 = fma(r61, r69, r67);
    r63 = r63 + r69;
    r65 = fma(r45, r63, r65);
    r63 = r40 * r65;
    r70 = r22 * r24;
    r70 = fma(r68, r25, r48 * r70);
    r70 = r70 + r69;
    r71 = r7 * r23;
    r71 = r71 * r61;
    r72 = r7 * r17;
    r72 = fma(r48, r72, r71);
    r48 = r7 * r20;
    r48 = r48 * r68;
    r73 = r66 * r43;
    r74 = r48 + r73;
    r75 = r72 + r74;
    r75 = fma(r6, r75, r5 * r70);
    r70 = r20 * r27;
    r70 = r70 * r66;
    r33 = r33 + r70;
    r75 = fma(r45, r33, r75);
    r49 = r49 * r49;
    r49 = 1.0 / r49;
    r49 = r4 * r49;
    r50 = r49 * r50;
    r63 = fma(r75, r50, r1 * r63);
    r33 = r75 * r49;
    r62 = r70 + r62;
    r59 = fma(r61, r43, r59);
    r70 = r7 * r17;
    r70 = fma(r68, r70, r64);
    r59 = r59 + r70;
    r59 = fma(r5, r59, r6 * r62);
    r62 = r20 * r24;
    r64 = r22 * r24;
    r64 = r64 * r66;
    r62 = fma(r68, r62, r64);
    r62 = r62 + r72;
    r59 = fma(r45, r62, r59);
    r62 = r0 * r59;
    r62 = fma(r1, r62, r34 * r33);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r63, r62);
    r33 = r7 * r23;
    r72 = r9 * r14;
    r76 = r13 * r10;
    r76 = fma(r58, r76, r42 * r72);
    r76 = fma(r12, r29, r76);
    r76 = fma(r42, r21, r76);
    r33 = r33 * r76;
    r21 = r8 * r11;
    r72 = r12 * r15;
    r72 = fma(r42, r72, r58 * r21);
    r72 = fma(r58, r16, r72);
    r72 = fma(r58, r19, r72);
    r19 = fma(r72, r43, r33);
    r19 = r19 + r69;
    r69 = fma(r61, r25, r64);
    r16 = r7 * r17;
    r16 = r16 * r72;
    r21 = r7 * r20;
    r21 = fma(r76, r21, r16);
    r69 = r69 + r21;
    r69 = fma(r5, r69, r6 * r19);
    r19 = r23 * r27;
    r19 = r19 * r66;
    r77 = r20 * r72;
    r78 = r27 * r77;
    r79 = r19 + r78;
    r69 = fma(r45, r79, r69);
    r79 = r76 * r60;
    r19 = r19 + r79;
    r73 = r71 + r73;
    r73 = r73 + r21;
    r73 = fma(r45, r73, r5 * r19);
    r19 = r17 * r24;
    r21 = r22 * r24;
    r21 = fma(r76, r21, r61 * r19);
    r19 = r7 * r23;
    r71 = r7 * r20;
    r71 = r71 * r66;
    r19 = fma(r72, r19, r71);
    r21 = r21 + r19;
    r73 = fma(r6, r21, r73);
    r21 = r40 * r73;
    r21 = fma(r1, r21, r69 * r50);
    r79 = r78 + r79;
    r33 = r67 + r33;
    r67 = r20 * r24;
    r33 = fma(r61, r67, r33);
    r78 = r22 * r24;
    r33 = fma(r72, r78, r33);
    r33 = fma(r45, r33, r6 * r79);
    r79 = r7 * r17;
    r76 = fma(r76, r43, r61 * r79);
    r76 = r76 + r19;
    r33 = fma(r5, r76, r33);
    r76 = r0 * r33;
    r79 = r69 * r49;
    r79 = fma(r34, r79, r1 * r76);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r21, r79);
    r76 = r23 * r27;
    r76 = r76 * r68;
    r60 = r66 * r60;
    r66 = r76 + r60;
    r64 = r48 + r64;
    r48 = r7 * r23;
    r61 = r8 * r14;
    r78 = r9 * r15;
    r78 = fma(r42, r78, r58 * r61);
    r61 = r12 * r10;
    r78 = fma(r42, r61, r78);
    r78 = fma(r13, r29, r78);
    r48 = r48 * r78;
    r29 = r17 * r24;
    r64 = fma(r72, r29, r64);
    r64 = r64 + r48;
    r64 = fma(r6, r64, r5 * r66);
    r66 = r7 * r17;
    r66 = r66 * r78;
    r29 = fma(r68, r43, r66);
    r29 = r29 + r19;
    r64 = fma(r45, r29, r64);
    r29 = r40 * r64;
    r43 = fma(r78, r43, r7 * r77);
    r43 = r43 + r70;
    r66 = r71 + r66;
    r71 = r22 * r24;
    r66 = fma(r68, r71, r66);
    r66 = fma(r72, r25, r66);
    r66 = fma(r5, r66, r6 * r43);
    r27 = r20 * r27;
    r27 = r27 * r78;
    r76 = r76 + r27;
    r66 = fma(r45, r76, r66);
    r29 = fma(r66, r50, r1 * r29);
    r76 = r66 * r49;
    r60 = r27 + r60;
    r27 = r22 * r24;
    r77 = fma(r24, r77, r78 * r27);
    r77 = r77 + r70;
    r77 = fma(r45, r77, r6 * r60);
    r48 = r16 + r48;
    r48 = r48 + r74;
    r77 = fma(r5, r48, r77);
    r48 = r0 * r77;
    r48 = fma(r1, r48, r34 * r76);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r29, r48);
    r76 = r40 * r30;
    r76 = fma(r36, r50, r1 * r76);
    r5 = r0 * r28;
    r74 = r36 * r49;
    r74 = fma(r34, r74, r1 * r5);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r76, r74);
    r5 = r40 * r35;
    r5 = fma(r37, r50, r1 * r5);
    r16 = r37 * r49;
    r45 = r0 * r32;
    r45 = fma(r1, r45, r34 * r16);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r5, r45);
    r16 = r40 * r51;
    r16 = fma(r1, r16, r31 * r50);
    r60 = r0 * r52;
    r6 = r31 * r49;
    r6 = fma(r34, r6, r1 * r60);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r16, r6);
    r60 = r4 * r3;
    r70 = r4 * r2;
    r70 = fma(r63, r70, r62 * r60);
    r60 = r4 * r2;
    r27 = r4 * r3;
    r27 = fma(r79, r27, r21 * r60);
    WriteSum2<double, double>((double*)inout_shared, r70, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r4 * r3;
    r70 = r4 * r2;
    r70 = fma(r29, r70, r48 * r27);
    r27 = r4 * r3;
    r60 = r4 * r2;
    r60 = fma(r76, r60, r74 * r27);
    WriteSum2<double, double>((double*)inout_shared, r70, r60);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r4 * r2;
    r70 = r4 * r3;
    r70 = fma(r45, r70, r5 * r60);
    r60 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r16, r27, r6 * r60);
    WriteSum2<double, double>((double*)inout_shared, r70, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r63, r63, r62 * r62);
    r70 = fma(r21, r21, r79 * r79);
    WriteSum2<double, double>((double*)inout_shared, r27, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r48, r48, r29 * r29);
    r27 = fma(r76, r76, r74 * r74);
    WriteSum2<double, double>((double*)inout_shared, r70, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r5, r5, r45 * r45);
    r70 = fma(r16, r16, r6 * r6);
    WriteSum2<double, double>((double*)inout_shared, r27, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r63, r21, r62 * r79);
    r27 = fma(r62, r48, r63 * r29);
    WriteSum2<double, double>((double*)inout_shared, r70, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r62, r74, r63 * r76);
    r70 = fma(r62, r45, r63 * r5);
    WriteSum2<double, double>((double*)inout_shared, r27, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fma(r62, r6, r63 * r16);
    r63 = fma(r79, r48, r21 * r29);
    WriteSum2<double, double>((double*)inout_shared, r62, r63);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fma(r21, r76, r79 * r74);
    r62 = fma(r79, r45, r21 * r5);
    WriteSum2<double, double>((double*)inout_shared, r63, r62);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fma(r21, r16, r79 * r6);
    r79 = fma(r29, r76, r48 * r74);
    WriteSum2<double, double>((double*)inout_shared, r21, r79);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fma(r29, r5, r48 * r45);
    r29 = fma(r29, r16, r48 * r6);
    WriteSum2<double, double>((double*)inout_shared, r79, r29);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r74, r45, r76 * r5);
    r76 = fma(r76, r16, r74 * r6);
    WriteSum2<double, double>((double*)inout_shared, r29, r76);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fma(r5, r16, r45 * r6);
    WriteSum1<double, double>((double*)inout_shared, r16);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r53 * r40;
    r16 = r16 * r41;
    r16 = fma(r1, r16, r50 * r39);
    r5 = r54 * r0;
    r5 = r5 * r41;
    r41 = r49 * r34;
    r41 = fma(r39, r41, r1 * r5);
    WriteIdx2<1024, double, double, double2>(
        out_sensor_from_rig_log_scale_jac,
        0 * out_sensor_from_rig_log_scale_jac_num_alloc,
        global_thread_idx,
        r16,
        r41);
    r5 = r4 * r3;
    r39 = r4 * r2;
    r39 = fma(r16, r39, r41 * r5);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_njtr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r39);
  if (global_thread_idx < problem_size) {
    r16 = fma(r16, r16, r41 * r41);
  };
  SumStore<double>(out_sensor_from_rig_log_scale_precond_diag_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r16);
  if (global_thread_idx < problem_size) {
    r16 = r40 * r46;
    r16 = fma(r26, r50, r1 * r16);
    r41 = r26 * r49;
    r39 = r0 * r56;
    r39 = fma(r1, r39, r34 * r41);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r16,
                                             r39);
    r41 = r40 * r57;
    r41 = fma(r44, r50, r1 * r41);
    r5 = r0 * r55;
    r6 = r44 * r49;
    r6 = fma(r34, r6, r1 * r5);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r41, r6);
    r5 = r40 * r18;
    r50 = fma(r47, r50, r1 * r5);
    r5 = r47 * r49;
    r45 = r0 * r38;
    r45 = fma(r1, r45, r34 * r5);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r50,
                                             r45);
    r5 = r4 * r2;
    r1 = r4 * r3;
    r1 = fma(r39, r1, r16 * r5);
    r5 = r4 * r2;
    r76 = r4 * r3;
    r76 = fma(r6, r76, r41 * r5);
    WriteSum2<double, double>((double*)inout_shared, r1, r76);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r4 * r2;
    r1 = r4 * r3;
    r1 = fma(r45, r1, r50 * r76);
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r39, r39, r16 * r16);
    r76 = fma(r6, r6, r41 * r41);
    WriteSum2<double, double>((double*)inout_shared, r1, r76);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fma(r50, r50, r45 * r45);
    WriteSum1<double, double>((double*)inout_shared, r76);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fma(r39, r6, r16 * r41);
    r39 = fma(r39, r45, r16 * r50);
    WriteSum2<double, double>((double*)inout_shared, r76, r39);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r6, r45, r41 * r50);
    WriteSum1<double, double>((double*)inout_shared, r45);
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