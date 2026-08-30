#include "colmap/estimators/point_refinement_caspar.h"

#include "colmap/estimators/caspar/caspar_model_adapter.h"
#include "colmap/estimators/caspar/row_reprojection_validation.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"

#include <chrono>
#include <utility>

#include <Eigen/Core>

namespace colmap {
namespace {

using Clock = std::chrono::steady_clock;

std::vector<StorageType> PackImageFromWorld(const float* matrices,
                                            const size_t num_images) {
  using InputProjection = Eigen::Matrix<float, 3, 4, Eigen::RowMajor>;
  using PackedProjection = Eigen::Matrix<StorageType, 3, 4>;

  std::vector<StorageType> packed(num_images * 12);
  for (size_t image_index = 0; image_index < num_images; ++image_index) {
    const Eigen::Map<const InputProjection> input(matrices + image_index * 12);
    Eigen::Map<PackedProjection> output(packed.data() + image_index * 12);
    output = input.cast<StorageType>();
  }
  return packed;
}

template <typename T>
std::vector<T> Convert(const float* data, const size_t size) {
  return std::vector<T>(data, data + size);
}

template <typename Finalize>
CasparPointRefinementResult RefineFixedCameraPinholePointsCasparImpl(
    const float* initial_points,
    const size_t num_points,
    const float* image_from_world,
    const size_t num_images,
    const uint32_t* observation_point_indices,
    const uint32_t* observation_image_indices,
    const float* observation_xy,
    const size_t num_observations,
    const CasparPointRefinementOptions& options,
    const size_t device_id,
    Finalize&& finalize) {
  CasparSolverSizing sizing;
  sizing.num_points = num_points;
  sizing.num_fixed_camera_pinhole_point = num_observations;
  sizing.num_fixed_camera_pinhole_point_images = num_images;
  auto solver =
      CreateSolver(CreateCasparSolverParameters(options), sizing, device_id);

  std::vector<StorageType> point_data =
      Convert<StorageType>(initial_points, num_points * 3);
  std::vector<StorageType> projection_data =
      PackImageFromWorld(image_from_world, num_images);
#ifdef CASPAR_USE_DOUBLE
  std::vector<StorageType> pixel_data =
      Convert<StorageType>(observation_xy, num_observations * 2);
  const StorageType* pixels = pixel_data.data();
#else
  const StorageType* pixels = observation_xy;
#endif
  solver.SetPointNodesFromStackedHost(point_data.data(), 0, num_points);
  solver.SetFixedCameraPinholePointNum(num_observations);
  solver.SetFixedCameraPinholePointPointIndicesFromHost(
      observation_point_indices, num_observations);
  solver.SetFixedCameraPinholePointImageFromWorldDataFromStackedHost(
      projection_data.data(), 0, num_images);
  solver.SetFixedCameraPinholePointImageFromWorldIndicesFromHost(
      observation_image_indices, num_observations);
  solver.SetFixedCameraPinholePointPixelDataFromStackedHost(
      pixels, 0, num_observations);
  solver.finish_indices();

  const caspar::SolveResult solve_result = solver.solve(
      /*print_progress=*/false,
      /*verbose_logging=*/options.collect_iteration_data);
  const double validation_seconds = finalize(solver);
  solver.GetPointNodesToStackedHost(point_data.data(), 0, num_points);

  CasparPointRefinementResult result;
#ifdef CASPAR_USE_DOUBLE
  result.points.reserve(point_data.size());
  for (const StorageType value : point_data) {
    result.points.push_back(static_cast<float>(value));
  }
#else
  result.points = std::move(point_data);
#endif
  result.summary = CasparBundleAdjustmentSummary::Create(solve_result);
  result.summary->num_residuals = static_cast<int>(2 * num_observations);
  result.summary->allocation_size = solver.get_allocation_size();
  result.validation_seconds = validation_seconds;
  return result;
}

}  // namespace

size_t SelectCasparDevice(const CasparSolverOptions& options) {
  const std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
  const int gpu_index = gpu_indices.front();
  return static_cast<size_t>(gpu_index >= 0 ? gpu_index : FindBestCudaDevice());
}

CasparPointRefinementResult RefineFixedCameraPinholePointsCaspar(
    const float* initial_points,
    const size_t num_points,
    const float* image_from_world,
    const size_t num_images,
    const uint32_t* observation_point_indices,
    const uint32_t* observation_image_indices,
    const float* observation_xy,
    const size_t num_observations,
    const CasparPointRefinementOptions& options) {
  return RefineFixedCameraPinholePointsCasparImpl(
      initial_points,
      num_points,
      image_from_world,
      num_images,
      observation_point_indices,
      observation_image_indices,
      observation_xy,
      num_observations,
      options,
      SelectCasparDevice(options),
      [](caspar::GraphSolver&) { return 0.0; });
}

CasparPointRefinementResult
RefineFixedCameraPinholePointsCasparWithReprojectionValidation(
    const float* initial_points,
    const size_t num_points,
    const float* image_from_world,
    const size_t num_images,
    const uint32_t* observation_point_indices,
    const uint32_t* observation_image_indices,
    const float* observation_xy,
    const size_t num_observations,
    const uint32_t* observation_offsets,
    const size_t row_point_start,
    const CasparPointRefinementOptions& options,
    CasparRowReprojectionValidator& validator) {
  return RefineFixedCameraPinholePointsCasparImpl(
      initial_points,
      num_points,
      image_from_world,
      num_images,
      observation_point_indices,
      observation_image_indices,
      observation_xy,
      num_observations,
      options,
      validator.DeviceId(),
      [&](caspar::GraphSolver& solver) {
        const Clock::time_point validation_start = Clock::now();
        validator.MeasureChunk(solver,
                               row_point_start,
                               num_points,
                               observation_offsets,
                               observation_image_indices,
                               observation_xy,
                               num_observations);
        return std::chrono::duration<double>(Clock::now() - validation_start)
            .count();
      });
}

}  // namespace colmap
