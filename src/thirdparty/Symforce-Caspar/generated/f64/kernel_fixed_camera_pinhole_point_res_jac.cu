#include "kernel_fixed_camera_pinhole_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) FixedCameraPinholePointResJacKernel(
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* image_from_world,
    unsigned int image_from_world_num_alloc,
    SharedIndex* image_from_world_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19;

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
    r6 = fma(r6, r16, r4);
    r6 = fma(r10, r18, r6);
    r6 = fma(r7, r13, r6);
    r1 = fma(r6, r3, r1 * r2);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r0, r1);
    r7 = r2 * r1;
    r15 = r15 * r15;
    r15 = 1.0 / r15;
    r15 = r2 * r15;
    r6 = r6 * r15;
    r16 = fma(r8, r6, r16 * r3);
    r10 = r2 * r0;
    r4 = r8 * r12;
    r5 = fma(r5, r3, r15 * r4);
    r10 = fma(r5, r10, r16 * r7);
    r7 = r2 * r1;
    r13 = fma(r14, r6, r13 * r3);
    r4 = r2 * r0;
    r19 = r14 * r12;
    r9 = fma(r9, r3, r15 * r19);
    r4 = fma(r9, r4, r13 * r7);
    WriteSum2<double, double>((double*)inout_shared, r10, r4);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = r2 * r1;
    r6 = fma(r11, r6, r18 * r3);
    r18 = r2 * r0;
    r10 = r11 * r12;
    r3 = fma(r17, r3, r15 * r10);
    r18 = fma(r3, r18, r6 * r4);
    WriteSum1<double, double>((double*)inout_shared, r18);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = fma(r16, r16, r5 * r5);
    r4 = fma(r13, r13, r9 * r9);
    WriteSum2<double, double>((double*)inout_shared, r18, r4);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = fma(r6, r6, r3 * r3);
    WriteSum1<double, double>((double*)inout_shared, r4);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = fma(r16, r13, r5 * r9);
    r16 = fma(r16, r6, r5 * r3);
    WriteSum2<double, double>((double*)inout_shared, r4, r16);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r13, r6, r9 * r3);
    WriteSum1<double, double>((double*)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void FixedCameraPinholePointResJac(
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* image_from_world,
    unsigned int image_from_world_num_alloc,
    SharedIndex* image_from_world_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
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
  FixedCameraPinholePointResJacKernel<<<n_blocks, 1024>>>(
      point,
      point_num_alloc,
      point_indices,
      image_from_world,
      image_from_world_num_alloc,
      image_from_world_indices,
      pixel,
      pixel_num_alloc,
      out_res,
      out_res_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar