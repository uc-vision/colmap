#include "colmap/estimators/caspar/row_reprojection_validation.h"

#ifdef CASPAR_USE_DOUBLE
#include "thirdparty/Symforce-Caspar/generated/f64/solver.h"
#else
#include "thirdparty/Symforce-Caspar/generated/f32/solver.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/transform_iterator.h>

namespace colmap {
namespace {

#ifdef CASPAR_USE_DOUBLE
using ValidationScalar = double;
#else
using ValidationScalar = float;
#endif

void CheckCuda(const cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

size_t SetCudaDevice(const size_t device_id) {
  CheckCuda(cudaSetDevice(static_cast<int>(device_id)));
  return device_id;
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const size_t size) {
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)));
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() { cudaFree(data_); }

  T* Data() { return data_; }
  const T* Data() const { return data_; }
  void CopyFromHost(const T* source, const size_t size) {
    CheckCuda(
        cudaMemcpy(data_, source, size * sizeof(T), cudaMemcpyHostToDevice));
  }

  void CopyToHost(T* target, const size_t size) const {
    CheckCuda(
        cudaMemcpy(target, data_, size * sizeof(T), cudaMemcpyDeviceToHost));
  }

 private:
  T* data_;
};

template <typename Scalar>
struct VectorTypes;

template <>
struct VectorTypes<float> {
  using Vector3 = float3;
  using Vector4 = float4;
};

template <>
struct VectorTypes<double> {
  using Vector3 = double3;
  using Vector4 = double4;
};

struct ProjectionMatrix3x4f {
  float4 rows[3];

  template <typename Scalar>
  __device__ typename VectorTypes<Scalar>::Vector3 Transform(
      const typename VectorTypes<Scalar>::Vector4& point) const {
    using Vector3 = typename VectorTypes<Scalar>::Vector3;
    return Vector3{
        static_cast<Scalar>(rows[0].x) * point.x +
            static_cast<Scalar>(rows[0].y) * point.y +
            static_cast<Scalar>(rows[0].z) * point.z +
            static_cast<Scalar>(rows[0].w) * point.w,
        static_cast<Scalar>(rows[1].x) * point.x +
            static_cast<Scalar>(rows[1].y) * point.y +
            static_cast<Scalar>(rows[1].z) * point.z +
            static_cast<Scalar>(rows[1].w) * point.w,
        static_cast<Scalar>(rows[2].x) * point.x +
            static_cast<Scalar>(rows[2].y) * point.y +
            static_cast<Scalar>(rows[2].z) * point.z +
            static_cast<Scalar>(rows[2].w) * point.w,
    };
  }
};

static_assert(sizeof(ProjectionMatrix3x4f) == 12 * sizeof(float));

template <typename Scalar>
__device__ typename VectorTypes<Scalar>::Vector4 Homogeneous(
    const typename VectorTypes<Scalar>::Vector3& point) {
  return typename VectorTypes<Scalar>::Vector4{
      point.x, point.y, point.z, Scalar{1}};
}

template <typename Scalar>
__global__ void PointMeanReprojectionErrorKernel(
    const typename VectorTypes<Scalar>::Vector3* points,
    const ProjectionMatrix3x4f* image_from_world,
    const uint32_t* observation_offsets,
    const uint32_t* observation_image_indices,
    const float2* observation_xy,
    const size_t num_points,
    float* point_errors) {
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = 8;
  using WarpReduce = cub::WarpReduce<Scalar>;
  __shared__ typename WarpReduce::TempStorage reduction[kWarpsPerBlock];

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const size_t point_index = blockIdx.x * kWarpsPerBlock + warp;
  if (point_index >= num_points) {
    return;
  }

  const auto point = Homogeneous<Scalar>(points[point_index]);
  const uint32_t observation_start = observation_offsets[point_index];
  const uint32_t observation_end = observation_offsets[point_index + 1];
  Scalar error_sum = 0;
  for (uint32_t observation = observation_start + lane;
       observation < observation_end;
       observation += kWarpSize) {
    const auto projected =
        image_from_world[observation_image_indices[observation]]
            .template Transform<Scalar>(point);
    const float2 measured = observation_xy[observation];
    const Scalar residual_x =
        projected.x / projected.z - static_cast<Scalar>(measured.x);
    const Scalar residual_y =
        projected.y / projected.z - static_cast<Scalar>(measured.y);
    error_sum += sqrt(residual_x * residual_x + residual_y * residual_y);
  }
  error_sum = WarpReduce(reduction[warp]).Sum(error_sum);
  if (lane == 0) {
    point_errors[point_index] =
        static_cast<float>(error_sum / (observation_end - observation_start));
  }
}

struct FloatToDouble {
  __host__ __device__ double operator()(const float value) const {
    return static_cast<double>(value);
  }
};

struct PercentilePosition {
  size_t left;
  size_t right;
  double right_weight;
};

PercentilePosition Position(const size_t count, const double percentile) {
  const double index = percentile * static_cast<double>(count - 1);
  return PercentilePosition{
      static_cast<size_t>(std::floor(index)),
      static_cast<size_t>(std::ceil(index)),
      index - std::floor(index),
  };
}

__global__ void GatherPercentilesKernel(const float* sorted_errors,
                                        const size_t median_left,
                                        const size_t median_right,
                                        const size_t p95_left,
                                        const size_t p95_right,
                                        float4* values) {
  *values = make_float4(sorted_errors[median_left],
                        sorted_errors[median_right],
                        sorted_errors[p95_left],
                        sorted_errors[p95_right]);
}

double Interpolate(const float left,
                   const float right,
                   const double right_weight) {
  return (1.0 - right_weight) * static_cast<double>(left) +
         right_weight * static_cast<double>(right);
}

}  // namespace

struct CasparRowReprojectionValidator::Impl {
  Impl(const size_t num_points,
       const float* image_from_world,
       const size_t num_images,
       const size_t maximum_chunk_points,
       const size_t maximum_chunk_observations,
       const size_t device_id)
      : num_points(num_points),
        device_id(SetCudaDevice(device_id)),
        image_from_world(num_images),
        chunk_points(maximum_chunk_points * 3),
        observation_offsets(maximum_chunk_points + 1),
        observation_image_indices(maximum_chunk_observations),
        observation_xy(maximum_chunk_observations),
        point_errors(num_points) {
    this->image_from_world.CopyFromHost(
        reinterpret_cast<const ProjectionMatrix3x4f*>(image_from_world),
        num_images);
  }

  size_t num_points;
  size_t device_id;
  DeviceBuffer<ProjectionMatrix3x4f> image_from_world;
  DeviceBuffer<ValidationScalar> chunk_points;
  DeviceBuffer<uint32_t> observation_offsets;
  DeviceBuffer<uint32_t> observation_image_indices;
  DeviceBuffer<float2> observation_xy;
  DeviceBuffer<float> point_errors;
};

CasparRowReprojectionValidator::CasparRowReprojectionValidator(
    const size_t num_points,
    const float* image_from_world,
    const size_t num_images,
    const size_t maximum_chunk_points,
    const size_t maximum_chunk_observations,
    const size_t device_id)
    : impl_(std::make_unique<Impl>(num_points,
                                   image_from_world,
                                   num_images,
                                   maximum_chunk_points,
                                   maximum_chunk_observations,
                                   device_id)) {}

CasparRowReprojectionValidator::~CasparRowReprojectionValidator() = default;

size_t CasparRowReprojectionValidator::DeviceId() const {
  return impl_->device_id;
}

void CasparRowReprojectionValidator::MeasureChunk(
    caspar::GraphSolver& solver,
    const size_t row_point_start,
    const size_t num_points,
    const uint32_t* observation_offsets,
    const uint32_t* observation_image_indices,
    const float* observation_xy,
    const size_t num_observations) {
  CheckCuda(cudaSetDevice(static_cast<int>(impl_->device_id)));
  solver.GetPointNodesToStackedDevice(
      impl_->chunk_points.Data(), 0, num_points);
  impl_->observation_offsets.CopyFromHost(observation_offsets, num_points + 1);
  impl_->observation_image_indices.CopyFromHost(observation_image_indices,
                                                num_observations);
  impl_->observation_xy.CopyFromHost(
      reinterpret_cast<const float2*>(observation_xy), num_observations);

  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  const size_t blocks = (num_points + kWarpsPerBlock - 1) / kWarpsPerBlock;
  PointMeanReprojectionErrorKernel<ValidationScalar><<<blocks, kThreads>>>(
      reinterpret_cast<const VectorTypes<ValidationScalar>::Vector3*>(
          impl_->chunk_points.Data()),
      impl_->image_from_world.Data(),
      impl_->observation_offsets.Data(),
      impl_->observation_image_indices.Data(),
      impl_->observation_xy.Data(),
      num_points,
      impl_->point_errors.Data() + row_point_start);
  CheckCuda(cudaGetLastError());
  CheckCuda(cudaDeviceSynchronize());
}

CasparReprojectionErrorSummary CasparRowReprojectionValidator::Summarize(
    const size_t num_observations) {
  CheckCuda(cudaSetDevice(static_cast<int>(impl_->device_id)));
  DeviceBuffer<double> sum(1);
  DeviceBuffer<float> sorted_errors(impl_->num_points);
  const auto errors = thrust::make_transform_iterator(
      impl_->point_errors.Data(), FloatToDouble{});

  size_t reduce_storage_bytes = 0;
  CheckCuda(cub::DeviceReduce::Sum(
      nullptr, reduce_storage_bytes, errors, sum.Data(), impl_->num_points));
  size_t sort_storage_bytes = 0;
  CheckCuda(cub::DeviceRadixSort::SortKeys(nullptr,
                                           sort_storage_bytes,
                                           impl_->point_errors.Data(),
                                           sorted_errors.Data(),
                                           impl_->num_points));
  DeviceBuffer<uint8_t> storage(
      std::max(reduce_storage_bytes, sort_storage_bytes));
  CheckCuda(cub::DeviceReduce::Sum(storage.Data(),
                                   reduce_storage_bytes,
                                   errors,
                                   sum.Data(),
                                   impl_->num_points));
  CheckCuda(cub::DeviceRadixSort::SortKeys(storage.Data(),
                                           sort_storage_bytes,
                                           impl_->point_errors.Data(),
                                           sorted_errors.Data(),
                                           impl_->num_points));

  const PercentilePosition median = Position(impl_->num_points, 0.5);
  const PercentilePosition p95 = Position(impl_->num_points, 0.95);
  DeviceBuffer<float4> percentile_values(1);
  GatherPercentilesKernel<<<1, 1>>>(sorted_errors.Data(),
                                    median.left,
                                    median.right,
                                    p95.left,
                                    p95.right,
                                    percentile_values.Data());
  CheckCuda(cudaGetLastError());

  double error_sum;
  float4 values;
  sum.CopyToHost(&error_sum, 1);
  percentile_values.CopyToHost(&values, 1);
  return CasparReprojectionErrorSummary{
      impl_->num_points,
      num_observations,
      error_sum / static_cast<double>(impl_->num_points),
      Interpolate(values.x, values.y, median.right_weight),
      Interpolate(values.z, values.w, p95.right_weight),
  };
}

}  // namespace colmap
