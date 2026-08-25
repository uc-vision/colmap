#include "colmap/estimators/rig_calibration.h"

#include "pycolmap/estimators/rig_calibration_packed.h"
#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindRigCalibration(py::module& m) {
  IsPyceresAvailable();

  using Observation = RigCalibrationObservation;
  auto PyObservation = py::classh<Observation>(m, "RigCalibrationObservation")
                           .def(py::init<>())
                           .def_readwrite("frame_idx", &Observation::frame_idx)
                           .def_readwrite("camera_id", &Observation::camera_id)
                           .def_readwrite("xy", &Observation::xy);
  MakeDataclass(PyObservation);

  using Track = RigCalibrationTrack;
  auto PyTrack = py::classh<Track>(m, "RigCalibrationTrack")
                     .def(py::init<>())
                     .def_readwrite("xyz", &Track::xyz)
                     .def_readwrite("observations", &Track::observations);
  MakeDataclass(PyTrack);

  using DistancePrior = RigCalibrationDistancePrior;
  auto PyDistancePrior =
      py::classh<DistancePrior>(m, "RigCalibrationDistancePrior")
          .def(py::init<>())
          .def_readwrite("distance", &DistancePrior::distance)
          .def_readwrite("stddev", &DistancePrior::stddev);
  MakeDataclass(PyDistancePrior);

  using Group = RigCalibrationGroup;
  auto PyGroup = py::classh<Group>(m, "RigCalibrationGroup")
                     .def(py::init<>())
                     .def_readwrite("rigs_from_group", &Group::rigs_from_group)
                     .def_readwrite("tracks", &Group::tracks)
                     .def_readwrite("frame0_to_frame2_distance",
                                    &Group::frame0_to_frame2_distance);
  MakeDataclass(PyGroup);

  using Options = RigCalibrationOptions;
  auto PyOptions =
      py::classh<Options>(m, "RigCalibrationOptions")
          .def(py::init<>())
          .def_readwrite("refine_focal_length", &Options::refine_focal_length)
          .def_readwrite("refine_principal_point",
                         &Options::refine_principal_point)
          .def_readwrite("refine_distortion", &Options::refine_distortion)
          .def_readwrite("refine_sensor_from_rig",
                         &Options::refine_sensor_from_rig)
          .def_readwrite("ceres", &Options::ceres)
          .def_readwrite("distance_loss_function_type",
                         &Options::distance_loss_function_type)
          .def_readwrite("distance_loss_function_scale",
                         &Options::distance_loss_function_scale)
          .def_readwrite("print_summary", &Options::print_summary)
          .def("check", &Options::Check);
  MakeDataclass(PyOptions);

  using Observability = RigCalibrationObservability;
  auto PyObservability =
      py::classh<Observability>(m, "RigCalibrationObservability")
          .def(py::init<>())
          .def_readwrite("parameter_names", &Observability::parameter_names)
          .def_readwrite("marginal_information",
                         &Observability::marginal_information)
          .def_readwrite("marginal_covariance",
                         &Observability::marginal_covariance)
          .def_readwrite("standard_deviations",
                         &Observability::standard_deviations)
          .def_readwrite("normalized_information_eigenvalues",
                         &Observability::normalized_information_eigenvalues)
          .def_readwrite("rank", &Observability::rank)
          .def_readwrite("normalized_condition_number",
                         &Observability::normalized_condition_number)
          .def("is_full_rank", &Observability::IsFullRank);
  MakeDataclass(PyObservability);

  using Summary = RigCalibrationSummary;
  auto PySummary =
      py::classh<Summary, CeresBundleAdjustmentSummary>(m,
                                                        "RigCalibrationSummary")
          .def(py::init<>())
          .def_readwrite("num_groups", &Summary::num_groups)
          .def_readwrite("num_tracks", &Summary::num_tracks)
          .def_readwrite("num_observations", &Summary::num_observations)
          .def_readwrite("num_filtered_groups", &Summary::num_filtered_groups)
          .def_readwrite("num_filtered_observations",
                         &Summary::num_filtered_observations)
          .def_readwrite("num_invalid_observations",
                         &Summary::num_invalid_observations)
          .def_readwrite("reprojection_rmse", &Summary::reprojection_rmse)
          .def_readwrite("distance_prior_rmse", &Summary::distance_prior_rmse)
          .def_readwrite("reprojection_errors", &Summary::reprojection_errors)
          .def_readwrite("distance_prior_errors",
                         &Summary::distance_prior_errors)
          .def_readwrite("stage_summaries", &Summary::stage_summaries)
          .def_readwrite("observability", &Summary::observability);
  MakeDataclass(PySummary);

  py::classh<CeresRigCalibrator>(m, "CeresRigCalibrator")
      .def("solve",
           [](CeresRigCalibrator& self) {
             py::gil_scoped_release release;
             return self.Solve();
           })
      .def_property_readonly("problem", &CeresRigCalibrator::Problem)
      .def_property_readonly("options", &CeresRigCalibrator::Options);

  m.def("create_ceres_rig_calibrator",
        &CreateCeresRigCalibrator,
        "options"_a,
        "rig_id"_a,
        "groups"_a,
        "reconstruction"_a,
        py::keep_alive<0, 4>());

  BindRigCalibrationPacked(m);
}
