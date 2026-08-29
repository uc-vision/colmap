#include "kernel_fixed_camera_pinhole_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    FixedCameraPinholePointResJacFirstKernel(
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* image_from_world,
        unsigned int image_from_world_num_alloc,
        SharedIndex* image_from_world_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        const double* const reprojection_loss_scale,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        double* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex image_from_world_indices_loc[1024];
  image_from_world_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? image_from_world_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r0, r1);
    r2 = -1.00000000000000000e+00;
    r3 = 1.00000000000000008e-15;
  };
  LoadShared<2, double, double>(image_from_world,
                                10 * image_from_world_num_alloc,
                                image_from_world_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        image_from_world_indices_loc[threadIdx.x].target,
                        r4,
                        r5);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  LoadShared<2, double, double>(image_from_world,
                                2 * image_from_world_num_alloc,
                                image_from_world_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        image_from_world_indices_loc[threadIdx.x].target,
                        r8,
                        r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r5 = fma(r6, r8, r5);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r10);
  };
  __syncthreads();
  LoadShared<2, double, double>(image_from_world,
                                8 * image_from_world_num_alloc,
                                image_from_world_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        image_from_world_indices_loc[threadIdx.x].target,
                        r11,
                        r12);
  };
  __syncthreads();
  LoadShared<2, double, double>(image_from_world,
                                4 * image_from_world_num_alloc,
                                image_from_world_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        image_from_world_indices_loc[threadIdx.x].target,
                        r13,
                        r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r5 = fma(r10, r11, r5);
    r5 = fma(r7, r14, r5);
    r15 = copysign(1.0, r5);
    r15 = fma(r3, r15, r5);
    r3 = 1.0 / r15;
  };
  LoadShared<2, double, double>(image_from_world,
                                0 * image_from_world_num_alloc,
                                image_from_world_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        image_from_world_indices_loc[threadIdx.x].target,
                        r5,
                        r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = fma(r6, r5, r12);
  };
  LoadShared<2, double, double>(image_from_world,
                                6 * image_from_world_num_alloc,
                                image_from_world_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        image_from_world_indices_loc[threadIdx.x].target,
                        r17,
                        r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = fma(r7, r9, r12);
    r12 = fma(r10, r17, r12);
    r0 = fma(r12, r3, r0 * r2);
    r19 = 2.00000000000000000e+00;
    r20 = 1.00000000000000000e+00;
  };
  LoadUnique<1, double, double>(
      reprojection_loss_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r21);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r21 = r21 * r21;
    r21 = 1.0 / r21;
    r6 = fma(r6, r16, r4);
    r6 = fma(r10, r18, r6);
    r6 = fma(r7, r13, r6);
    r1 = fma(r6, r3, r1 * r2);
    r7 = fma(r1, r1, r0 * r0);
    r7 = fma(r7, r21, r20);
    r10 = sqrt(r7);
    r10 = r20 + r10;
    r20 = 1.0 / r10;
    r4 = r19 * r20;
    r22 = sqrt(r4);
    r23 = r0 * r22;
    r24 = r1 * r22;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r23, r24);
    r24 = r19 * r0;
    r24 = r24 * r0;
    r25 = r1 * r20;
    r26 = r19 * r1;
    r25 = fma(r26, r25, r20 * r24);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r25);
  if (global_thread_idx < problem_size) {
    r25 = r2 * r1;
    r24 = r19 * r0;
    r15 = r15 * r15;
    r15 = 1.0 / r15;
    r15 = r2 * r15;
    r12 = r12 * r15;
    r5 = fma(r5, r3, r8 * r12);
    r27 = r8 * r6;
    r27 = fma(r15, r27, r16 * r3);
    r24 = fma(r27, r26, r5 * r24);
    r16 = -5.00000000000000000e-01;
    r21 = r16 * r21;
    r7 = rsqrt(r7);
    r4 = rsqrt(r4);
    r10 = r10 * r10;
    r10 = 1.0 / r10;
    r21 = r21 * r7;
    r21 = r21 * r4;
    r21 = r21 * r10;
    r24 = r24 * r21;
    r27 = fma(r27, r22, r1 * r24);
    r25 = r25 * r27;
    r23 = r2 * r23;
    r24 = fma(r0, r24, r5 * r22);
    r25 = fma(r24, r23, r22 * r25);
    r9 = fma(r9, r3, r14 * r12);
    r5 = r19 * r0;
    r10 = r14 * r6;
    r10 = fma(r15, r10, r13 * r3);
    r5 = fma(r10, r26, r9 * r5);
    r13 = r0 * r5;
    r13 = fma(r21, r13, r9 * r22);
    r9 = r2 * r1;
    r4 = r1 * r5;
    r10 = fma(r10, r22, r21 * r4);
    r9 = r9 * r10;
    r9 = fma(r22, r9, r13 * r23);
    WriteSum2<double, double>((double*)inout_shared, r25, r9);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = fma(r17, r3, r11 * r12);
    r12 = r19 * r0;
    r9 = r11 * r6;
    r9 = fma(r15, r9, r18 * r3);
    r26 = fma(r9, r26, r17 * r12);
    r12 = r0 * r26;
    r12 = fma(r21, r12, r17 * r22);
    r17 = r2 * r1;
    r3 = r1 * r26;
    r9 = fma(r9, r22, r21 * r3);
    r17 = r17 * r9;
    r17 = fma(r22, r17, r12 * r23);
    WriteSum1<double, double>((double*)inout_shared, r17);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = fma(r27, r27, r24 * r24);
    r23 = fma(r13, r13, r10 * r10);
    WriteSum2<double, double>((double*)inout_shared, r17, r23);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r12, r12, r9 * r9);
    WriteSum1<double, double>((double*)inout_shared, r23);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r24, r13, r27 * r10);
    r27 = fma(r27, r9, r24 * r12);
    WriteSum2<double, double>((double*)inout_shared, r23, r27);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fma(r10, r9, r13 * r12);
    WriteSum1<double, double>((double*)inout_shared, r9);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void FixedCameraPinholePointResJacFirst(
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* image_from_world,
    unsigned int image_from_world_num_alloc,
    SharedIndex* image_from_world_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    const double* const reprojection_loss_scale,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
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
  FixedCameraPinholePointResJacFirstKernel<<<n_blocks, 1024>>>(
      point,
      point_num_alloc,
      point_indices,
      image_from_world,
      image_from_world_num_alloc,
      image_from_world_indices,
      pixel,
      pixel_num_alloc,
      reprojection_loss_scale,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar