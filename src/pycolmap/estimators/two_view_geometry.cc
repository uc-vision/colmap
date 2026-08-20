#include "colmap/estimators/two_view_geometry.h"

#include "colmap/feature/types.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/homography_matrix.h"
#include "colmap/geometry/normalization.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

using FeaturePointsMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

void BindTwoViewGeometryEstimator(py::module& m) {
  py::classh<TwoViewGeometryOptions> PyTwoViewGeometryOptions(
      m, "TwoViewGeometryOptions");
  PyTwoViewGeometryOptions.def(py::init<>())
      .def_readwrite("min_num_inliers",
                     &TwoViewGeometryOptions::min_num_inliers)
      .def_readwrite("min_inlier_ratio",
                     &TwoViewGeometryOptions::min_inlier_ratio)
      .def_readwrite("min_E_F_inlier_ratio",
                     &TwoViewGeometryOptions::min_E_F_inlier_ratio)
      .def_readwrite("max_H_inlier_ratio",
                     &TwoViewGeometryOptions::max_H_inlier_ratio)
      .def_readwrite("watermark_min_inlier_ratio",
                     &TwoViewGeometryOptions::watermark_min_inlier_ratio)
      .def_readwrite("watermark_border_size",
                     &TwoViewGeometryOptions::watermark_border_size)
      .def_readwrite("detect_watermark",
                     &TwoViewGeometryOptions::detect_watermark)
      .def_readwrite("multiple_ignore_watermark",
                     &TwoViewGeometryOptions::multiple_ignore_watermark)
      .def_readwrite("watermark_detection_max_error",
                     &TwoViewGeometryOptions::watermark_detection_max_error)
      .def_readwrite("filter_stationary_matches",
                     &TwoViewGeometryOptions::filter_stationary_matches)
      .def_readwrite("stationary_matches_max_error",
                     &TwoViewGeometryOptions::stationary_matches_max_error)
      .def_readwrite("force_H_use", &TwoViewGeometryOptions::force_H_use)
      .def_readwrite("compute_relative_pose",
                     &TwoViewGeometryOptions::compute_relative_pose)
      .def_readwrite("multiple_models",
                     &TwoViewGeometryOptions::multiple_models)
      .def_readwrite("ransac", &TwoViewGeometryOptions::ransac_options);
  MakeDataclass(PyTwoViewGeometryOptions);

  auto PyFixedRigMatchedPair =
      py::classh<FixedRigMatchedPair>(m, "FixedRigMatchedPair")
          .def(py::init<>())
          .def(py::init(
                   [](const image_t image_id1,
                      const image_t image_id2,
                      const Camera& camera1,
                      const Camera& camera2,
                      const Eigen::Ref<const FeatureMatchesMatrix>& matches,
                      const Eigen::Ref<const FeaturePointsMatrix>& points1,
                      const Eigen::Ref<const FeaturePointsMatrix>& points2) {
                     FixedRigMatchedPair pair;
                     pair.image_id1 = image_id1;
                     pair.image_id2 = image_id2;
                     pair.camera1 = camera1;
                     pair.camera2 = camera2;
                     pair.matches = MatchesFromMatrix(matches);
                     pair.points1.resize(points1.rows());
                     pair.points2.resize(points2.rows());
                     Eigen::Map<FeaturePointsMatrix>(
                         reinterpret_cast<double*>(pair.points1.data()),
                         points1.rows(),
                         2) = points1;
                     Eigen::Map<FeaturePointsMatrix>(
                         reinterpret_cast<double*>(pair.points2.data()),
                         points2.rows(),
                         2) = points2;
                     return pair;
                   }),
               "image_id1"_a,
               "image_id2"_a,
               "camera1"_a,
               "camera2"_a,
               "matches"_a,
               "points1"_a,
               "points2"_a)
          .def_readwrite("image_id1", &FixedRigMatchedPair::image_id1)
          .def_readwrite("image_id2", &FixedRigMatchedPair::image_id2)
          .def_readwrite("camera1", &FixedRigMatchedPair::camera1)
          .def_readwrite("camera2", &FixedRigMatchedPair::camera2)
          .def_property(
              "matches",
              [](const FixedRigMatchedPair& self) {
                return MatchesToMatrix(self.matches);
              },
              [](FixedRigMatchedPair& self,
                 const Eigen::Ref<const FeatureMatchesMatrix>& matches) {
                self.matches = MatchesFromMatrix(matches);
              })
          .def_readwrite("points1", &FixedRigMatchedPair::points1)
          .def_readwrite("points2", &FixedRigMatchedPair::points2);
  MakeDataclass(PyFixedRigMatchedPair);

  m.def(
      "estimate_fixed_rig_two_view_geometries",
      [](const Rig& rig1,
         const Rig& rig2,
         const std::vector<const FixedRigMatchedPair*>& pair_refs,
         const TwoViewGeometryOptions& options,
         const size_t max_num_ransac_matches) {
        py::gil_scoped_release release;
        std::vector<FixedRigMatchedPair> pairs;
        pairs.reserve(pair_refs.size());
        for (const FixedRigMatchedPair* pair : pair_refs) {
          pairs.push_back(*pair);
        }
        auto geometries = EstimateFixedRigTwoViewGeometries(
            rig1, rig2, pairs, options, max_num_ransac_matches);
        std::vector<std::tuple<image_t, image_t, TwoViewGeometry>> result;
        result.reserve(geometries.size());
        for (auto& [image_pair, geometry] : geometries) {
          result.emplace_back(
              image_pair.first, image_pair.second, std::move(geometry));
        }
        return result;
      },
      "rig1"_a,
      "rig2"_a,
      "pairs"_a,
      py::arg_v(
          "options", TwoViewGeometryOptions(), "TwoViewGeometryOptions()"),
      "max_num_ransac_matches"_a = 4096,
      "Estimate one fixed-rig relative pose without accessing a database.");

  m.def(
      "estimate_calibrated_two_view_geometry",
      [](const Camera& camera1,
         const std::vector<Eigen::Vector2d>& points1,
         const Camera& camera2,
         const std::vector<Eigen::Vector2d>& points2,
         const FeatureMatchesMatrix* matches_ptr,
         const TwoViewGeometryOptions& options) {
        py::gil_scoped_release release;
        FeatureMatches matches;
        if (matches_ptr != nullptr) {
          matches = MatchesFromMatrix(*matches_ptr);
        } else {
          THROW_CHECK_EQ(points1.size(), points2.size());
          matches.reserve(points1.size());
          for (size_t i = 0; i < points1.size(); i++) {
            matches.emplace_back(i, i);
          }
        }
        return EstimateCalibratedTwoViewGeometry(
            camera1, points1, camera2, points2, matches, options);
      },
      "camera1"_a,
      "points1"_a,
      "camera2"_a,
      "points2"_a,
      "matches"_a = py::none(),
      py::arg_v(
          "options", TwoViewGeometryOptions(), "TwoViewGeometryOptions()"));

  m.def(
      "estimate_two_view_geometry",
      [](const Camera& camera1,
         const std::vector<Eigen::Vector2d>& points1,
         const Camera& camera2,
         const std::vector<Eigen::Vector2d>& points2,
         const FeatureMatchesMatrix* matches_ptr,
         const TwoViewGeometryOptions& options) {
        py::gil_scoped_release release;
        FeatureMatches matches;
        if (matches_ptr != nullptr) {
          matches = MatchesFromMatrix(*matches_ptr);
        } else {
          THROW_CHECK_EQ(points1.size(), points2.size());
          matches.reserve(points1.size());
          for (size_t i = 0; i < points1.size(); i++) {
            matches.emplace_back(i, i);
          }
        }
        return EstimateTwoViewGeometry(
            camera1, points1, camera2, points2, std::move(matches), options);
      },
      "camera1"_a,
      "points1"_a,
      "camera2"_a,
      "points2"_a,
      "matches"_a = py::none(),
      py::arg_v(
          "options", TwoViewGeometryOptions(), "TwoViewGeometryOptions()"));

  m.def("estimate_two_view_geometry_pose",
        &EstimateTwoViewGeometryPose,
        "camera1"_a,
        "points1"_a,
        "camera2"_a,
        "points2"_a,
        "geometry"_a);

  m.def(
      "compute_squared_sampson_error",
      [](const std::vector<Eigen::Vector2d>& points1,
         const std::vector<Eigen::Vector2d>& points2,
         const Eigen::Matrix3d& E) {
        std::vector<double> residuals;
        ComputeSquaredSampsonError(points1, points2, E, &residuals);
        return residuals;
      },
      "points2D1"_a,
      "points2D2"_a,
      "E"_a,
      "Calculate the squared Sampson error for a given essential or "
      "fundamental matrix.",
      py::call_guard<py::gil_scoped_release>());
}
