#include "colmap/estimators/point_refinement_caspar.h"

#include "colmap/estimators/caspar/caspar_model_adapter.h"
#include "colmap/util/cuda.h"

#include <utility>

#include <Eigen/Core>

namespace colmap {
namespace {

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

}  // namespace

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
  caspar::SolverParams<StorageType> parameters;
  parameters.solver_iter_max = options.solver_iter_max;
  parameters.pcg_iter_max = options.pcg_iter_max;

  CasparSolverSizing sizing;
  sizing.num_points = num_points;
  sizing.num_fixed_camera_pinhole_point = num_observations;
  sizing.num_fixed_camera_pinhole_point_images = num_images;
  const size_t device_id = static_cast<size_t>(
      options.gpu_index >= 0 ? options.gpu_index : FindBestCudaDevice());
  auto solver = CreateSolver(parameters, sizing, device_id);

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
  const StorageType loss_scale = options.loss_scale;

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
  solver.SetFixedCameraPinholePointReprojectionLossScaleDataFromStackedHost(
      &loss_scale);
  solver.finish_indices();

  const caspar::SolveResult solve_result = solver.solve(
      /*print_progress=*/false,
      /*verbose_logging=*/false);
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
  return result;
}

}  // namespace colmap
