#include "kernel_fixed_camera_pinhole_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) FixedCameraPinholePointScoreKernel(
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* image_from_world,
    unsigned int image_from_world_num_alloc,
    SharedIndex* image_from_world_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    const double* const reprojection_loss_scale,
    double* const out_rTr,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13;

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
    r8 = fma(r6, r8, r5);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r5);
  };
  __syncthreads();
  LoadShared<2, double, double>(image_from_world,
                                8 * image_from_world_num_alloc,
                                image_from_world_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        image_from_world_indices_loc[threadIdx.x].target,
                        r10,
                        r11);
  };
  __syncthreads();
  LoadShared<2, double, double>(image_from_world,
                                4 * image_from_world_num_alloc,
                                image_from_world_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        image_from_world_indices_loc[threadIdx.x].target,
                        r12,
                        r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = fma(r5, r10, r8);
    r8 = fma(r7, r13, r8);
    r13 = copysign(1.0, r8);
    r13 = fma(r3, r13, r8);
    r13 = 1.0 / r13;
  };
  LoadShared<2, double, double>(image_from_world,
                                0 * image_from_world_num_alloc,
                                image_from_world_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        image_from_world_indices_loc[threadIdx.x].target,
                        r3,
                        r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = fma(r6, r3, r11);
  };
  LoadShared<2, double, double>(image_from_world,
                                6 * image_from_world_num_alloc,
                                image_from_world_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        image_from_world_indices_loc[threadIdx.x].target,
                        r11,
                        r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = fma(r7, r9, r3);
    r3 = fma(r5, r11, r3);
    r3 = fma(r3, r13, r0 * r2);
    r3 = r3 * r3;
    r0 = 2.00000000000000000e+00;
    r11 = 1.00000000000000000e+00;
  };
  LoadUnique<1, double, double>(
      reprojection_loss_scale, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = r9 * r9;
    r9 = 1.0 / r9;
    r8 = fma(r6, r8, r4);
    r8 = fma(r5, r10, r8);
    r8 = fma(r7, r12, r8);
    r13 = fma(r8, r13, r1 * r2);
    r13 = r13 * r13;
    r8 = r3 + r13;
    r9 = fma(r8, r9, r11);
    r9 = sqrt(r9);
    r9 = r11 + r9;
    r9 = 1.0 / r9;
    r9 = r0 * r9;
    r9 = fma(r13, r9, r3 * r9);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r9);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void FixedCameraPinholePointScore(double* point,
                                  unsigned int point_num_alloc,
                                  SharedIndex* point_indices,
                                  double* image_from_world,
                                  unsigned int image_from_world_num_alloc,
                                  SharedIndex* image_from_world_indices,
                                  double* pixel,
                                  unsigned int pixel_num_alloc,
                                  const double* const reprojection_loss_scale,
                                  double* const out_rTr,
                                  size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  FixedCameraPinholePointScoreKernel<<<n_blocks, 1024>>>(
      point,
      point_num_alloc,
      point_indices,
      image_from_world,
      image_from_world_num_alloc,
      image_from_world_indices,
      pixel,
      pixel_num_alloc,
      reprojection_loss_scale,
      out_rTr,
      problem_size);
}

}  // namespace caspar