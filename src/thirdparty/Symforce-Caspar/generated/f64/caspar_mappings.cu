#include "caspar_mappings.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// We use shared memory to improve the memory access.
// A smaller block size of 32 allows for larger nodetypes.
constexpr int block_size = 32;

namespace caspar {

__global__
__launch_bounds__(block_size, 1) void ConstImageFromWorldStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 12];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 12 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 12;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 12] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 12;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[6];
    data[1] = stacked_local_ptr[7];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[8];
    data[1] = stacked_local_ptr[9];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 8 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[10];
    data[1] = stacked_local_ptr[11];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 10 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstImageFromWorldCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 12];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 12;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[6] = data[0];
    stacked_local_ptr[7] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 8 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[8] = data[0];
    stacked_local_ptr[9] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 10 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[10] = data[0];
    stacked_local_ptr[11] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 12 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 12;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 12];
  }
}

cudaError_t ConstImageFromWorldStackedToCaspar(const double* stacked_data,
                                               double* cas_data,
                                               const unsigned int cas_stride,
                                               const unsigned int cas_offset,
                                               const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstImageFromWorldStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstImageFromWorldCasparToStacked(const double* cas_data,
                                               double* stacked_data,
                                               const unsigned int cas_stride,
                                               const unsigned int cas_offset,
                                               const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstImageFromWorldCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholeCalibStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 4];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 4] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholeCalibCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 4];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 4];
  }
}

cudaError_t ConstPinholeCalibStackedToCaspar(const double* stacked_data,
                                             double* cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeCalibStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPinholeCalibCasparToStacked(const double* cas_data,
                                             double* stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeCalibCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholeFocalStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholeFocalCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 2];
  }
}

cudaError_t ConstPinholeFocalStackedToCaspar(const double* stacked_data,
                                             double* cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeFocalStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPinholeFocalCasparToStacked(const double* cas_data,
                                             double* stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeFocalCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholePoseStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 7] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[6];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholePoseCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[6] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 7];
  }
}

cudaError_t ConstPinholePoseStackedToCaspar(const double* stacked_data,
                                            double* cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholePoseStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPinholePoseCasparToStacked(const double* cas_data,
                                            double* stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholePoseCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholePrincipalPointStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholePrincipalPointCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 2];
  }
}

cudaError_t ConstPinholePrincipalPointStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholePrincipalPointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPinholePrincipalPointCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholePrincipalPointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholeSensorFromRigStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 7] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[6];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholeSensorFromRigCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[6] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 7];
  }
}

cudaError_t ConstPinholeSensorFromRigStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeSensorFromRigStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPinholeSensorFromRigCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeSensorFromRigCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPixelStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPixelCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 2];
  }
}

cudaError_t ConstPixelStackedToCaspar(const double* stacked_data,
                                      double* cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPixelStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPixelCasparToStacked(const double* cas_data,
                                      double* stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPixelCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPointStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 3] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPointCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[2] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 3];
  }
}

cudaError_t ConstPointStackedToCaspar(const double* stacked_data,
                                      double* cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPointCasparToStacked(const double* cas_data,
                                      double* stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPositionSqrtInformationStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 9];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 9 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 9;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 9] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 9;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[6];
    data[1] = stacked_local_ptr[7];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[8];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 8 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPositionSqrtInformationCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 9];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 9;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[6] = data[0];
    stacked_local_ptr[7] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 8 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[8] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 9 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 9;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 9];
  }
}

cudaError_t ConstPositionSqrtInformationStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPositionSqrtInformationStackedToCaspar_kernel<<<num_blocks,
                                                       block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPositionSqrtInformationCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPositionSqrtInformationCasparToStacked_kernel<<<num_blocks,
                                                       block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstReferencePositionStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 3] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstReferencePositionCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[2] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 3];
  }
}

cudaError_t ConstReferencePositionStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstReferencePositionStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstReferencePositionCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstReferencePositionCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstReprojectionLossScaleStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 1];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 1 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 1;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 1] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 1;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstReprojectionLossScaleCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 1];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 1;
    const double* in_ptr;
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[0] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 1 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 1;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 1];
  }
}

cudaError_t ConstReprojectionLossScaleStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstReprojectionLossScaleStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstReprojectionLossScaleCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstReprojectionLossScaleCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialFocalAndExtraStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialFocalAndExtraCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 2];
  }
}

cudaError_t ConstSimpleRadialFocalAndExtraStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialFocalAndExtraStackedToCaspar_kernel<<<num_blocks,
                                                         block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstSimpleRadialFocalAndExtraCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialFocalAndExtraCasparToStacked_kernel<<<num_blocks,
                                                         block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialPoseStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 7] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[6];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialPoseCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[6] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 7];
  }
}

cudaError_t ConstSimpleRadialPoseStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialPoseStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstSimpleRadialPoseCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialPoseCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialPrincipalPointStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialPrincipalPointCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 2];
  }
}

cudaError_t ConstSimpleRadialPrincipalPointStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialPrincipalPointStackedToCaspar_kernel<<<num_blocks,
                                                          block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstSimpleRadialPrincipalPointCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialPrincipalPointCasparToStacked_kernel<<<num_blocks,
                                                          block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialSensorFromRigStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 7] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[6];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialSensorFromRigCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[6] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 7];
  }
}

cudaError_t ConstSimpleRadialSensorFromRigStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialSensorFromRigStackedToCaspar_kernel<<<num_blocks,
                                                         block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstSimpleRadialSensorFromRigCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialSensorFromRigCasparToStacked_kernel<<<num_blocks,
                                                         block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void PinholeCalibStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 4];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 4] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void PinholeCalibCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 4];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 4];
  }
}

cudaError_t PinholeCalibStackedToCaspar(const double* stacked_data,
                                        double* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholeCalibStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PinholeCalibCasparToStacked(const double* cas_data,
                                        double* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholeCalibCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void PinholeFocalStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void PinholeFocalCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 2];
  }
}

cudaError_t PinholeFocalStackedToCaspar(const double* stacked_data,
                                        double* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholeFocalStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PinholeFocalCasparToStacked(const double* cas_data,
                                        double* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholeFocalCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void PinholePoseStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 7] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[6];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void PinholePoseCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[6] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 7];
  }
}

cudaError_t PinholePoseStackedToCaspar(const double* stacked_data,
                                       double* cas_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholePoseStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PinholePoseCasparToStacked(const double* cas_data,
                                       double* stacked_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholePoseCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void PinholePrincipalPointStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void PinholePrincipalPointCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 2];
  }
}

cudaError_t PinholePrincipalPointStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholePrincipalPointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PinholePrincipalPointCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholePrincipalPointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__ __launch_bounds__(block_size, 1) void PointStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 3] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void PointCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[2] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 3];
  }
}

cudaError_t PointStackedToCaspar(const double* stacked_data,
                                 double* cas_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PointCasparToStacked(const double* cas_data,
                                 double* stacked_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void SensorFromRigLogScaleStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 1];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 1 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 1;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 1] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 1;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void SensorFromRigLogScaleCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 1];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 1;
    const double* in_ptr;
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[0] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 1 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 1;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 1];
  }
}

cudaError_t SensorFromRigLogScaleStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SensorFromRigLogScaleStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t SensorFromRigLogScaleCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SensorFromRigLogScaleCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialCalibStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 4];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 4] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialCalibCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 4];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 4];
  }
}

cudaError_t SimpleRadialCalibStackedToCaspar(const double* stacked_data,
                                             double* cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialCalibStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t SimpleRadialCalibCasparToStacked(const double* cas_data,
                                             double* stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialCalibCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialFocalAndExtraStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialFocalAndExtraCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 2];
  }
}

cudaError_t SimpleRadialFocalAndExtraStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialFocalAndExtraStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t SimpleRadialFocalAndExtraCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialFocalAndExtraCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialPoseStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 7] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[6];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialPoseCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[6] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 7];
  }
}

cudaError_t SimpleRadialPoseStackedToCaspar(const double* stacked_data,
                                            double* cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialPoseStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t SimpleRadialPoseCasparToStacked(const double* cas_data,
                                            double* stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialPoseCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialPrincipalPointStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialPrincipalPointCasparToStacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 2];
  }
}

cudaError_t SimpleRadialPrincipalPointStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialPrincipalPointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t SimpleRadialPrincipalPointCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialPrincipalPointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

}  // namespace caspar