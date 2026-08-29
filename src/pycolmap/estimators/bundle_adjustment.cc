#include "colmap/estimators/bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#ifdef CASPAR_ENABLED
#include "colmap/estimators/point_refinement_caspar.h"
#endif

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <algorithm>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

#ifdef CASPAR_ENABLED
using FloatArray = py::array_t<float, py::array::c_style>;
using Uint32Array = py::array_t<uint32_t, py::array::c_style>;

struct PyCasparPointRefinementResult {
  py::array_t<float> points;
  std::shared_ptr<CasparBundleAdjustmentSummary> summary;
};

py::array_t<float> PointArray(std::vector<float>&& values,
                              const size_t num_points) {
  auto* storage = new std::vector<float>(std::move(values));
  py::capsule owner(storage, [](void* pointer) {
    delete static_cast<std::vector<float>*>(pointer);
  });
  return py::array_t<float>(
      {static_cast<ssize_t>(num_points), static_cast<ssize_t>(3)},
      {static_cast<ssize_t>(3 * sizeof(float)),
       static_cast<ssize_t>(sizeof(float))},
      storage->data(),
      owner);
}

PyCasparPointRefinementResult RefineFixedCameraPinholePoints(
    const FloatArray& initial_points,
    const FloatArray& image_from_world,
    const Uint32Array& observation_point_indices,
    const Uint32Array& observation_image_indices,
    const FloatArray& observation_xy,
    const CasparPointRefinementOptions& options) {
  THROW_CHECK_EQ(initial_points.ndim(), 2);
  THROW_CHECK_EQ(initial_points.shape(1), 3);
  THROW_CHECK_GT(initial_points.shape(0), 0);
  THROW_CHECK_EQ(image_from_world.ndim(), 3);
  THROW_CHECK_EQ(image_from_world.shape(1), 3);
  THROW_CHECK_EQ(image_from_world.shape(2), 4);
  THROW_CHECK_GT(image_from_world.shape(0), 0);
  THROW_CHECK_EQ(observation_point_indices.ndim(), 1);
  THROW_CHECK_EQ(observation_image_indices.ndim(), 1);
  THROW_CHECK_EQ(observation_xy.ndim(), 2);
  THROW_CHECK_EQ(observation_xy.shape(1), 2);
  const size_t num_observations = observation_point_indices.shape(0);
  THROW_CHECK_GT(num_observations, 0);
  THROW_CHECK_EQ(observation_image_indices.shape(0), num_observations);
  THROW_CHECK_EQ(observation_xy.shape(0), num_observations);

  const size_t num_points = initial_points.shape(0);
  const size_t num_images = image_from_world.shape(0);
  CasparPointRefinementResult result;
  {
    py::gil_scoped_release release;
    THROW_CHECK_LT(
        *std::max_element(observation_point_indices.data(),
                          observation_point_indices.data() + num_observations),
        num_points);
    THROW_CHECK_LT(
        *std::max_element(observation_image_indices.data(),
                          observation_image_indices.data() + num_observations),
        num_images);
    result =
        RefineFixedCameraPinholePointsCaspar(initial_points.data(),
                                             num_points,
                                             image_from_world.data(),
                                             num_images,
                                             observation_point_indices.data(),
                                             observation_image_indices.data(),
                                             observation_xy.data(),
                                             num_observations,
                                             options);
  }
  return {PointArray(std::move(result.points), num_points),
          std::move(result.summary)};
}
#endif

class PyBundleAdjuster : public BundleAdjuster,
                         py::trampoline_self_life_support {
 public:
  PyBundleAdjuster(const BundleAdjustmentOptions& options,
                   const BundleAdjustmentConfig& config)
      : BundleAdjuster(options, config) {}

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    PYBIND11_OVERRIDE_PURE(
        std::shared_ptr<BundleAdjustmentSummary>, BundleAdjuster, Solve);
  }
};

class PyCeresBundleAdjuster : public CeresBundleAdjuster,
                              py::trampoline_self_life_support {
 public:
  PyCeresBundleAdjuster(const BundleAdjustmentOptions& options,
                        const BundleAdjustmentConfig& config)
      : CeresBundleAdjuster(options, config) {}

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    PYBIND11_OVERRIDE_PURE(
        std::shared_ptr<BundleAdjustmentSummary>, CeresBundleAdjuster, Solve);
  }

  std::shared_ptr<ceres::Problem>& Problem() override {
    // Cannot use PYBIND11_OVERRIDE_PURE for reference returns as it creates
    // a temporary. Instead, manually call override and store in member.
    py::gil_scoped_acquire gil;
    py::function override = py::get_override(
        static_cast<const CeresBundleAdjuster*>(this), "problem");
    if (override) {
      auto obj = override();
      problem_ = obj.cast<std::shared_ptr<ceres::Problem>>();
    }
    return problem_;
  }

 private:
  std::shared_ptr<ceres::Problem> problem_;
};

}  // namespace

void BindBundleAdjuster(py::module& m) {
  IsPyceresAvailable();  // Try to import pyceres to populate the docstrings.

  auto PyBundleAdjustmentTerminationType =
      py::enum_<BundleAdjustmentTerminationType>(
          m, "BundleAdjustmentTerminationType")
          .value("CONVERGENCE", BundleAdjustmentTerminationType::CONVERGENCE)
          .value("NO_CONVERGENCE",
                 BundleAdjustmentTerminationType::NO_CONVERGENCE)
          .value("FAILURE", BundleAdjustmentTerminationType::FAILURE)
          .value("USER_SUCCESS", BundleAdjustmentTerminationType::USER_SUCCESS)
          .value("USER_FAILURE", BundleAdjustmentTerminationType::USER_FAILURE);
  AddStringToEnumConstructor(PyBundleAdjustmentTerminationType);

  using BASummary = BundleAdjustmentSummary;
  auto PyBundleAdjustmentSummary =
      py::classh<BASummary>(m, "BundleAdjustmentSummary")
          .def(py::init<>())
          .def_readwrite("termination_type", &BASummary::termination_type)
          .def_readwrite("num_residuals", &BASummary::num_residuals)
          .def("is_solution_usable", &BASummary::IsSolutionUsable)
          .def("brief_report", &BASummary::BriefReport);
  MakeDataclass(PyBundleAdjustmentSummary);

  using CeresBASummary = CeresBundleAdjustmentSummary;
  auto PyCeresBundleAdjustmentSummary =
      py::classh<CeresBASummary, BASummary>(m, "CeresBundleAdjustmentSummary")
          .def(py::init<>())
          .def_readwrite("ceres_summary",
                         &CeresBASummary::ceres_summary,
                         "Full Ceres solver summary.");
  MakeDataclass(PyCeresBundleAdjustmentSummary);

  using FixedRigPosePriorBASummary = FixedRigPosePriorBundleAdjustmentSummary;
  auto PyFixedRigPosePriorBundleAdjustmentSummary =
      py::classh<FixedRigPosePriorBASummary, BASummary>(
          m, "FixedRigPosePriorBundleAdjustmentSummary")
          .def(py::init<>())
          .def_readwrite("sensor_from_rig_scale",
                         &FixedRigPosePriorBASummary::sensor_from_rig_scale,
                         "Multiplicative correction applied to the current "
                         "sensor-from-rig translations.");
  MakeDataclass(PyFixedRigPosePriorBundleAdjustmentSummary);

#ifdef CASPAR_ENABLED
  using CasparBASummary = CasparBundleAdjustmentSummary;
  auto PyCasparBundleAdjustmentSummary =
      py::classh<CasparBASummary, BASummary>(m, "CasparBundleAdjustmentSummary")
          .def(py::init<>())
          .def_readwrite("iteration_count", &CasparBASummary::iteration_count)
          .def_readwrite("initial_score", &CasparBASummary::initial_score)
          .def_readwrite("final_score", &CasparBASummary::final_score)
          .def_readwrite("runtime", &CasparBASummary::runtime)
          .def_readwrite("allocation_size", &CasparBASummary::allocation_size);
  MakeDataclass(PyCasparBundleAdjustmentSummary);

  using CasparPointOpts = CasparPointRefinementOptions;
  auto PyCasparPointRefinementOptions =
      py::classh<CasparPointOpts>(m, "CasparPointRefinementOptions")
          .def(py::init<>())
          .def_readwrite("solver_iter_max", &CasparPointOpts::solver_iter_max)
          .def_readwrite("pcg_iter_max", &CasparPointOpts::pcg_iter_max)
          .def_readwrite("gpu_index", &CasparPointOpts::gpu_index)
          .def_readwrite("loss_scale", &CasparPointOpts::loss_scale);
  MakeDataclass(PyCasparPointRefinementOptions);

  py::classh<PyCasparPointRefinementResult>(m, "CasparPointRefinementResult")
      .def_readonly("points", &PyCasparPointRefinementResult::points)
      .def_readonly("summary", &PyCasparPointRefinementResult::summary);
#endif

  auto PyBundleAdjustmentGauge =
      py::enum_<BundleAdjustmentGauge>(m, "BundleAdjustmentGauge")
          .value("UNSPECIFIED", BundleAdjustmentGauge::UNSPECIFIED)
          .value("TWO_CAMS_FROM_WORLD",
                 BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD)
          .value("THREE_POINTS", BundleAdjustmentGauge::THREE_POINTS);
  AddStringToEnumConstructor(PyBundleAdjustmentGauge);

  auto PyBundleAdjustmentBackend =
      py::enum_<BundleAdjustmentBackend>(m, "BundleAdjustmentBackend")
          .value("CERES", BundleAdjustmentBackend::CERES)
          .value("CASPAR", BundleAdjustmentBackend::CASPAR)
          .value("CASPAR_RIG_SCHUR", BundleAdjustmentBackend::CASPAR_RIG_SCHUR);
  AddStringToEnumConstructor(PyBundleAdjustmentBackend);
  m.def("is_bundle_adjustment_backend_available",
        &IsBundleAdjustmentBackendAvailable,
        "backend"_a);

  using BACfg = BundleAdjustmentConfig;
  py::classh<BACfg> PyBundleAdjustmentConfig(m, "BundleAdjustmentConfig");
  PyBundleAdjustmentConfig.def(py::init<>())
      .def("fix_gauge", &BACfg::FixGauge)
      .def_property_readonly("fixed_gauge", &BACfg::FixedGauge)
      .def("num_points", &BACfg::NumPoints)
      .def("num_constant_cam_intrinsics", &BACfg::NumConstantCamIntrinsics)
      .def("num_constant_sensor_from_rig_poses",
           &BACfg::NumConstantSensorFromRigPoses)
      .def("num_constant_rig_from_world_poses",
           &BACfg::NumConstantRigFromWorldPoses)
      .def("num_variable_points", &BACfg::NumVariablePoints)
      .def("num_constant_points", &BACfg::NumConstantPoints)
      .def("num_residuals", &BACfg::NumResiduals, "reconstruction"_a)
      .def("add_image", &BACfg::AddImage, "image_id"_a)
      .def("has_image", &BACfg::HasImage, "image_id"_a)
      .def("remove_image", &BACfg::RemoveImage, "image_id"_a)
      .def("set_constant_cam_intrinsics",
           &BACfg::SetConstantCamIntrinsics,
           "camera_id"_a)
      .def("set_variable_cam_intrinsics",
           &BACfg::SetVariableCamIntrinsics,
           "camera_id"_a)
      .def("has_constant_cam_intrinsics",
           &BACfg::HasConstantCamIntrinsics,
           "camera_id"_a)
      .def("set_constant_sensor_from_rig_pose",
           &BACfg::SetConstantSensorFromRigPose,
           "sensor_id"_a)
      .def("set_variable_sensor_from_rig_pose",
           &BACfg::SetVariableSensorFromRigPose,
           "sensor_id"_a)
      .def("has_constant_sensor_from_rig_pose",
           &BACfg::HasConstantSensorFromRigPose,
           "sensor_id"_a)
      .def("set_constant_rig_from_world_pose",
           &BACfg::SetConstantRigFromWorldPose,
           "frame_id"_a)
      .def("set_variable_rig_from_world_pose",
           &BACfg::SetVariableRigFromWorldPose,
           "frame_id"_a)
      .def("has_constant_rig_from_world_pose",
           &BACfg::HasConstantRigFromWorldPose,
           "frame_id"_a)
      .def("add_variable_point", &BACfg::AddVariablePoint, "point3D_id"_a)
      .def("add_constant_point", &BACfg::AddConstantPoint, "point3D_id"_a)
      .def("has_point", &BACfg::HasPoint, "point3D_id"_a)
      .def("has_variable_point", &BACfg::HasVariablePoint, "point3D_id"_a)
      .def("has_constant_point", &BACfg::HasConstantPoint, "point3D_id"_a)
      .def("remove_variable_point", &BACfg::RemoveVariablePoint, "point3D_id"_a)
      .def("remove_constant_point", &BACfg::RemoveConstantPoint, "point3D_id"_a)
      .def_property_readonly("constant_cam_intrinsics",
                             &BACfg::ConstantCamIntrinsics)
      .def_property_readonly("images", &BACfg::Images)
      .def_property_readonly("variable_points", &BACfg::VariablePoints)
      .def_property_readonly("constant_points", &BACfg::ConstantPoints)
      .def_property_readonly("constant_sensor_from_rig_poses",
                             &BACfg::ConstantSensorFromRigPoses)
      .def_property_readonly("constant_rig_from_world_poses",
                             &BACfg::ConstantRigFromWorldPoses);
  MakeDataclass(PyBundleAdjustmentConfig);

  // Ceres-specific bundle adjustment options
  using CeresBAOpts = CeresBundleAdjustmentOptions;
  auto PyCeresLossFunctionType =
      py::enum_<CeresBAOpts::LossFunctionType>(m, "LossFunctionType")
          .value("TRIVIAL", CeresBAOpts::LossFunctionType::TRIVIAL)
          .value("SOFT_L1", CeresBAOpts::LossFunctionType::SOFT_L1)
          .value("CAUCHY", CeresBAOpts::LossFunctionType::CAUCHY)
          .value("HUBER", CeresBAOpts::LossFunctionType::HUBER);
  AddStringToEnumConstructor(PyCeresLossFunctionType);

  auto PyCeresBundleAdjustmentOptions =
      py::classh<CeresBAOpts>(m, "CeresBundleAdjustmentOptions")
          .def(py::init<>())
          .def("create_loss_function", &CeresBAOpts::CreateLossFunction)
          .def("create_solver_options",
               py::overload_cast<const BundleAdjustmentConfig&,
                                 const ceres::Problem&>(
                   &CeresBAOpts::CreateSolverOptions, py::const_),
               "config"_a,
               "problem"_a)
          .def_readwrite("loss_function_type",
                         &CeresBAOpts::loss_function_type,
                         "Loss function types: Trivial (non-robust) and Cauchy "
                         "(robust) loss.")
          .def_readwrite("loss_function_scale",
                         &CeresBAOpts::loss_function_scale,
                         "Scaling factor determines residual at which "
                         "robustification takes place.")
          .def_readwrite("use_gpu",
                         &CeresBAOpts::use_gpu,
                         "Whether to use Ceres' CUDA linear algebra library, "
                         "if available.")
          .def_readwrite("gpu_index",
                         &CeresBAOpts::gpu_index,
                         "Which GPU to use for solving the problem.")
          .def_readwrite("solver_options",
                         &CeresBAOpts::solver_options,
                         "Options for the Ceres solver. Using this member "
                         "requires having PyCeres installed.")
          .def_readwrite("min_num_images_gpu_solver",
                         &CeresBAOpts::min_num_images_gpu_solver,
                         "Minimum number of images to use the GPU solver.")
          .def_readwrite(
              "min_num_residuals_for_cpu_multi_threading",
              &CeresBAOpts::min_num_residuals_for_cpu_multi_threading,
              "Minimum number of residuals to enable multi-threading. Note "
              "that single-threaded is typically better for small bundle "
              "adjustment problems due to the overhead of threading.")
          .def_readwrite("max_num_images_direct_dense_cpu_solver",
                         &CeresBAOpts::max_num_images_direct_dense_cpu_solver,
                         "Threshold to switch between direct, sparse, and "
                         "iterative solvers.")
          .def_readwrite("max_num_images_direct_sparse_cpu_solver",
                         &CeresBAOpts::max_num_images_direct_sparse_cpu_solver,
                         "Threshold to switch between direct, sparse, and "
                         "iterative solvers.")
          .def_readwrite("max_num_images_direct_dense_gpu_solver",
                         &CeresBAOpts::max_num_images_direct_dense_gpu_solver,
                         "Threshold to switch between direct, sparse, and "
                         "iterative solvers.")
          .def_readwrite("max_num_images_direct_sparse_gpu_solver",
                         &CeresBAOpts::max_num_images_direct_sparse_gpu_solver,
                         "Threshold to switch between direct, sparse, and "
                         "iterative solvers.")
          .def_readwrite(
              "auto_select_solver_type",
              &CeresBAOpts::auto_select_solver_type,
              "Whether to automatically select solver type based on "
              "problem size. When False, uses the linear_solver_type "
              "and preconditioner_type from solver_options directly.")
          .def("check", &CeresBAOpts::Check);
  MakeDataclass(PyCeresBundleAdjustmentOptions);

  // Caspar-specific bundle adjustment options
  using CasparBAOpts = CasparBundleAdjustmentOptions;
  auto PyCasparBundleAdjustmentOptions =
      py::classh<CasparBAOpts>(m, "CasparBundleAdjustmentOptions")
          .def(py::init<>())
          .def_readwrite("solver_iter_max",
                         &CasparBAOpts::solver_iter_max,
                         "Maximum number of Caspar solver iterations.")
          .def_readwrite("pcg_iter_max",
                         &CasparBAOpts::pcg_iter_max,
                         "Maximum number of PCG iterations per solver step.")
          .def_readwrite("diag_init",
                         &CasparBAOpts::diag_init,
                         "Initial diagonal damping value.")
          .def_readwrite("diag_min",
                         &CasparBAOpts::diag_min,
                         "Minimum diagonal damping value.")
          .def_readwrite("diag_scaling_up",
                         &CasparBAOpts::diag_scaling_up,
                         "Diagonal damping increase factor.")
          .def_readwrite("diag_scaling_down",
                         &CasparBAOpts::diag_scaling_down,
                         "Diagonal damping decrease factor.")
          .def_readwrite("diag_exit_value",
                         &CasparBAOpts::diag_exit_value,
                         "Diagonal damping value that triggers termination.")
          .def_readwrite("score_exit_value",
                         &CasparBAOpts::score_exit_value,
                         "Score threshold that triggers termination.")
          .def_readwrite("pcg_rel_error_exit",
                         &CasparBAOpts::pcg_rel_error_exit,
                         "Relative PCG error threshold that triggers exit.")
          .def_readwrite("pcg_rel_score_exit",
                         &CasparBAOpts::pcg_rel_score_exit,
                         "Relative PCG score threshold that triggers exit.")
          .def_readwrite("pcg_rel_decrease_min",
                         &CasparBAOpts::pcg_rel_decrease_min,
                         "Minimum relative PCG decrease.")
          .def_readwrite("solver_rel_decrease_min",
                         &CasparBAOpts::solver_rel_decrease_min,
                         "Minimum relative solver decrease.")
          .def_readwrite("fixed_rig_reprojection_loss_scale",
                         &CasparBAOpts::fixed_rig_reprojection_loss_scale,
                         "Pseudo-Huber loss scale in pixels for fixed-rig "
                         "reprojection residuals.")
          .def_readwrite("gpu_index",
                         &CasparBAOpts::gpu_index,
                         "Which GPU to use for solving the problem.");
  MakeDataclass(PyCasparBundleAdjustmentOptions);

  // Solver-agnostic bundle adjustment options
  using BAOpts = BundleAdjustmentOptions;
  auto PyBundleAdjustmentOptions =
      py::classh<BAOpts>(m, "BundleAdjustmentOptions")
          .def(py::init<>())
          .def_readwrite("refine_focal_length",
                         &BAOpts::refine_focal_length,
                         "Whether to refine the focal length parameter group.")
          .def_readwrite(
              "refine_principal_point",
              &BAOpts::refine_principal_point,
              "Whether to refine the principal point parameter group.")
          .def_readwrite("refine_extra_params",
                         &BAOpts::refine_extra_params,
                         "Whether to refine the extra parameter group.")
          .def_readwrite("refine_rig_from_world",
                         &BAOpts::refine_rig_from_world,
                         "Whether to refine the frame from world extrinsic "
                         "parameter group.")
          .def_readwrite("refine_sensor_from_rig",
                         &BAOpts::refine_sensor_from_rig,
                         "Whether to refine the sensor from rig extrinsic "
                         "parameter group.")
          .def_readwrite("constant_rig_from_world_rotation",
                         &BAOpts::constant_rig_from_world_rotation,
                         "Whether to keep the rotation component of "
                         "rig_from_world constant. Only takes effect when "
                         "refine_rig_from_world is true.")
          .def_readwrite("refine_points3D",
                         &BAOpts::refine_points3D,
                         "Whether to refine 3D points.")
          .def_readwrite("min_track_length",
                         &BAOpts::min_track_length,
                         "Minimum track length for a 3D point.")
          .def_readwrite("print_summary",
                         &BAOpts::print_summary,
                         "Whether to print a final summary.")
          .def_readwrite("backend",
                         &BAOpts::backend,
                         "Solver backend to use for bundle adjustment.")
          .def_readwrite("ceres",
                         &BAOpts::ceres,
                         "Ceres-specific bundle adjustment options.")
          .def_readwrite("caspar",
                         &BAOpts::caspar,
                         "Caspar-specific bundle adjustment options.")
          .def("check", &BAOpts::Check);
  MakeDataclass(PyBundleAdjustmentOptions);

  // Ceres-specific pose prior bundle adjustment options
  using CeresPosePriorBAOpts = CeresPosePriorBundleAdjustmentOptions;
  auto PyCeresPosePriorBundleAdjustmentOptions =
      py::classh<CeresPosePriorBAOpts>(m,
                                       "CeresPosePriorBundleAdjustmentOptions")
          .def(py::init<>())
          .def_readwrite(
              "prior_position_loss_function_type",
              &CeresPosePriorBAOpts::prior_position_loss_function_type,
              "Loss function for prior position loss.")
          .def_readwrite("prior_position_loss_scale",
                         &CeresPosePriorBAOpts::prior_position_loss_scale,
                         "Threshold on the residual for the robust loss (chi2 "
                         "for 3DOF at 95% = 7.815).")
          .def("check", &CeresPosePriorBAOpts::Check);
  MakeDataclass(PyCeresPosePriorBundleAdjustmentOptions);

  // Solver-agnostic pose prior bundle adjustment options
  using PosePriorBAOpts = PosePriorBundleAdjustmentOptions;
  auto PyPosePriorBundleAdjustmentOptions =
      py::classh<PosePriorBAOpts>(m, "PosePriorBundleAdjustmentOptions")
          .def(py::init<>())
          .def_readwrite(
              "prior_position_fallback_stddev",
              &PosePriorBAOpts::prior_position_fallback_stddev,
              "Fallback if no prior position covariance is provided.")
          .def_readwrite("alignment_ransac",
                         &PosePriorBAOpts::alignment_ransac_options,
                         "RANSAC options for Sim3 alignment.")
          .def_readwrite("ceres",
                         &PosePriorBAOpts::ceres,
                         "Ceres-specific pose prior bundle adjustment options.")
          .def("check", &PosePriorBAOpts::Check);
  MakeDataclass(PyPosePriorBundleAdjustmentOptions);

  using FixedRigPosePriorBAOpts = FixedRigPosePriorBundleAdjustmentOptions;
  auto PyFixedRigPosePriorBundleAdjustmentOptions =
      py::classh<FixedRigPosePriorBAOpts>(
          m, "FixedRigPosePriorBundleAdjustmentOptions")
          .def(py::init<>())
          .def_readwrite(
              "prior_position_fallback_stddev",
              &FixedRigPosePriorBAOpts::prior_position_fallback_stddev,
              "Fallback standard deviation if a reference-camera "
              "position covariance is absent.")
          .def_readwrite("min_track_length",
                         &FixedRigPosePriorBAOpts::min_track_length,
                         "Minimum track length for a 3D point.")
          .def_readwrite("print_summary",
                         &FixedRigPosePriorBAOpts::print_summary,
                         "Whether to print a final summary.")
          .def_readwrite("backend",
                         &FixedRigPosePriorBAOpts::backend,
                         "Solver backend to use for bundle adjustment.")
          .def_readwrite("ceres",
                         &FixedRigPosePriorBAOpts::ceres,
                         "Ceres-specific bundle adjustment options.")
          .def_readwrite("caspar",
                         &FixedRigPosePriorBAOpts::caspar,
                         "Caspar-specific bundle adjustment options.")
          .def("check", &FixedRigPosePriorBAOpts::Check);
  MakeDataclass(PyFixedRigPosePriorBundleAdjustmentOptions);

  py::classh<BundleAdjuster, PyBundleAdjuster>(m, "BundleAdjuster")
      .def(py::init([](const BundleAdjustmentOptions& options,
                       const BundleAdjustmentConfig& config) {
             return new PyBundleAdjuster(options, config);
           }),
           "options"_a,
           "config"_a)
      .def("solve", &BundleAdjuster::Solve)
      .def_property_readonly("options", &BundleAdjuster::Options)
      .def_property_readonly("config", &BundleAdjuster::Config);

  py::classh<CeresBundleAdjuster, BundleAdjuster, PyCeresBundleAdjuster>(
      m, "CeresBundleAdjuster")
      .def(py::init([](const BundleAdjustmentOptions& options,
                       const BundleAdjustmentConfig& config) {
             return new PyCeresBundleAdjuster(options, config);
           }),
           "options"_a,
           "config"_a)
      .def_property_readonly("problem", &CeresBundleAdjuster::Problem);

  m.def("create_default_bundle_adjuster",
        CreateDefaultBundleAdjuster,
        "options"_a,
        "config"_a,
        "reconstruction"_a);

  m.def("create_default_ceres_bundle_adjuster",
        CreateDefaultCeresBundleAdjuster,
        "options"_a,
        "config"_a,
        "reconstruction"_a);

  m.def("create_pose_prior_bundle_adjuster",
        CreatePosePriorBundleAdjuster,
        "options"_a,
        "prior_options"_a,
        "config"_a,
        "pose_priors"_a,
        "reconstruction"_a);

  m.def("create_pose_prior_ceres_bundle_adjuster",
        CreatePosePriorCeresBundleAdjuster,
        "options"_a,
        "prior_options"_a,
        "config"_a,
        "pose_priors"_a,
        "reconstruction"_a);

  m.def("fixed_rig_pose_prior_bundle_adjustment",
        FixedRigPosePriorBundleAdjustment,
        "reconstruction"_a,
        "pose_priors"_a,
        "options"_a = FixedRigPosePriorBundleAdjustmentOptions(),
        "Jointly optimize rig-frame poses and 3D points with reference-camera "
        "position priors, a first-frame rotation gauge anchor, and one shared "
        "positive correction on non-reference sensor-from-rig translations.");
#ifdef CASPAR_ENABLED
  m.def("caspar_refine_pinhole_points",
        RefineFixedCameraPinholePoints,
        "initial_points"_a.noconvert(),
        "image_from_world"_a.noconvert(),
        "observation_point_indices"_a.noconvert(),
        "observation_image_indices"_a.noconvert(),
        "observation_xy"_a.noconvert(),
        "options"_a = CasparPointRefinementOptions(),
        "Refine 3D points against fixed distortion-free 3x4 projection "
        "matrices with CASPAR.");
#endif
}
