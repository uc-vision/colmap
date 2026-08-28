#include "pycolmap/scene/reconstruction.h"

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction_io.h"
#include "colmap/sensor/models.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/ply.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"

#include <filesystem>
#include <memory>
#include <sstream>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

py::dict TrackArrays(const Reconstruction& reconstruction) {
  const ssize_t num_points = static_cast<ssize_t>(reconstruction.NumPoints3D());
  const ssize_t num_observations =
      static_cast<ssize_t>(reconstruction.ComputeNumObservations());
  py::array_t<int64_t> point3D_ids(num_points);
  py::array_t<float> point_xyz(std::vector<ssize_t>{num_points, 3});
  py::array_t<uint8_t> point_rgb(std::vector<ssize_t>{num_points, 3});
  py::array_t<double> point_errors(num_points);
  py::array_t<int64_t> observation_offsets(num_points + 1);
  py::array_t<image_t> observation_image_ids(num_observations);
  py::array_t<point2D_t> observation_point2D_indices(num_observations);
  py::array_t<float> observation_xy(std::vector<ssize_t>{num_observations, 2});

  int64_t* point3D_ids_ptr = point3D_ids.mutable_data();
  float* point_xyz_ptr = point_xyz.mutable_data();
  uint8_t* point_rgb_ptr = point_rgb.mutable_data();
  double* point_errors_ptr = point_errors.mutable_data();
  int64_t* observation_offsets_ptr = observation_offsets.mutable_data();
  image_t* observation_image_ids_ptr = observation_image_ids.mutable_data();
  point2D_t* observation_point2D_indices_ptr =
      observation_point2D_indices.mutable_data();
  float* observation_xy_ptr = observation_xy.mutable_data();

  {
    py::gil_scoped_release release;
    ssize_t point_index = 0;
    ssize_t observation_index = 0;
    observation_offsets_ptr[0] = 0;
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      point3D_ids_ptr[point_index] = static_cast<int64_t>(point3D_id);
      for (int index = 0; index < 3; ++index) {
        point_xyz_ptr[3 * point_index + index] =
            static_cast<float>(point3D.xyz[index]);
        point_rgb_ptr[3 * point_index + index] = point3D.color[index];
      }
      point_errors_ptr[point_index] = point3D.error;
      for (const TrackElement& element : point3D.track.Elements()) {
        const Eigen::Vector2d& xy = reconstruction.Image(element.image_id)
                                        .Point2D(element.point2D_idx)
                                        .xy;
        observation_image_ids_ptr[observation_index] = element.image_id;
        observation_point2D_indices_ptr[observation_index] =
            element.point2D_idx;
        observation_xy_ptr[2 * observation_index] = static_cast<float>(xy.x());
        observation_xy_ptr[2 * observation_index + 1] =
            static_cast<float>(xy.y());
        ++observation_index;
      }
      observation_offsets_ptr[++point_index] = observation_index;
    }
  }

  return py::dict(
      "point3D_ids"_a = std::move(point3D_ids),
      "point_xyz"_a = std::move(point_xyz),
      "point_rgb"_a = std::move(point_rgb),
      "point_errors"_a = std::move(point_errors),
      "observation_offsets"_a = std::move(observation_offsets),
      "observation_image_ids"_a = std::move(observation_image_ids),
      "observation_point2D_indices"_a = std::move(observation_point2D_indices),
      "observation_xy"_a = std::move(observation_xy));
}

py::array_t<point3D_t> AddPoints3DFromArrays(
    Reconstruction& reconstruction,
    const py::array_t<float, py::array::c_style>& point_xyz,
    const py::array_t<uint8_t, py::array::c_style>& point_rgb,
    const py::array_t<int64_t, py::array::c_style>& observation_offsets,
    const py::array_t<image_t, py::array::c_style>& observation_image_ids,
    const py::array_t<point2D_t, py::array::c_style>&
        observation_point2D_indices) {
  THROW_CHECK_EQ(point_xyz.ndim(), 2);
  THROW_CHECK_EQ(point_xyz.shape(1), 3);
  const ssize_t num_points = point_xyz.shape(0);
  THROW_CHECK_EQ(point_rgb.ndim(), 2);
  THROW_CHECK_EQ(point_rgb.shape(0), num_points);
  THROW_CHECK_EQ(point_rgb.shape(1), 3);
  THROW_CHECK_EQ(observation_offsets.ndim(), 1);
  THROW_CHECK_EQ(observation_offsets.shape(0), num_points + 1);
  THROW_CHECK_EQ(observation_image_ids.ndim(), 1);
  THROW_CHECK_EQ(observation_point2D_indices.ndim(), 1);
  THROW_CHECK_EQ(observation_image_ids.shape(0),
                 observation_point2D_indices.shape(0));

  const auto xyz = point_xyz.unchecked<2>();
  const auto rgb = point_rgb.unchecked<2>();
  const auto offsets = observation_offsets.unchecked<1>();
  const auto image_ids = observation_image_ids.unchecked<1>();
  const auto point2D_indices = observation_point2D_indices.unchecked<1>();
  THROW_CHECK_EQ(offsets(0), 0);
  THROW_CHECK_EQ(offsets(num_points), observation_image_ids.shape(0));

  py::array_t<point3D_t> point3D_ids(num_points);
  point3D_t* point3D_ids_ptr = point3D_ids.mutable_data();
  {
    py::gil_scoped_release release;
    for (ssize_t point_index = 0; point_index < num_points; ++point_index) {
      const int64_t start = offsets(point_index);
      const int64_t end = offsets(point_index + 1);
      THROW_CHECK_LE(start, end);
      Track track;
      track.Reserve(static_cast<size_t>(end - start));
      for (int64_t observation_index = start; observation_index < end;
           ++observation_index) {
        track.AddElement(image_ids(observation_index),
                         point2D_indices(observation_index));
      }
      point3D_ids_ptr[point_index] = reconstruction.AddPoint3D(
          Eigen::Vector3d(xyz(point_index, 0),
                          xyz(point_index, 1),
                          xyz(point_index, 2)),
          std::move(track),
          Eigen::Vector3ub(rgb(point_index, 0),
                           rgb(point_index, 1),
                           rgb(point_index, 2)));
    }
  }
  return point3D_ids;
}

void BindReconstruction(py::module& m) {
  py::classh<Reconstruction>(m, "Reconstruction")
      .def(py::init<>())
      .def(py::init<const Reconstruction&>(), "reconstruction"_a)
      .def(py::init([](const std::filesystem::path& path) {
             auto reconstruction = std::make_shared<Reconstruction>();
             reconstruction->Read(path);
             return reconstruction;
           }),
           "path"_a)
      .def("read",
           &Reconstruction::Read,
           "path"_a,
           "Read reconstruction in COLMAP format. Prefer binary.")
      .def("write",
           &Reconstruction::Write,
           "output_dir"_a,
           "Write reconstruction in COLMAP binary format.")
      .def("read_text", &Reconstruction::ReadText, "path"_a)
      .def("read_binary", &Reconstruction::ReadBinary, "path"_a)
      .def("write_text", &Reconstruction::WriteText, "path"_a)
      .def("write_binary", &Reconstruction::WriteBinary, "path"_a)
      .def("num_rigs", &Reconstruction::NumRigs)
      .def("num_cameras", &Reconstruction::NumCameras)
      .def("num_frames", &Reconstruction::NumFrames)
      .def("num_reg_frames", &Reconstruction::NumRegFrames)
      .def("num_images", &Reconstruction::NumImages)
      .def("num_reg_images", &Reconstruction::NumRegImages)
      .def("num_points3D", &Reconstruction::NumPoints3D)
      .def_property_readonly("rigs",
                             &Reconstruction::Rigs,
                             py::return_value_policy::reference_internal)
      .def("rig",
           py::overload_cast<rig_t>(&Reconstruction::Rig),
           "rig_id"_a,
           "Direct accessor for a rig.",
           py::return_value_policy::reference_internal)
      .def_property_readonly("cameras",
                             &Reconstruction::Cameras,
                             py::return_value_policy::reference_internal)
      .def("camera",
           py::overload_cast<camera_t>(&Reconstruction::Camera),
           "camera_id"_a,
           "Direct accessor for a camera.",
           py::return_value_policy::reference_internal)
      .def_property_readonly("frames",
                             &Reconstruction::Frames,
                             py::return_value_policy::reference_internal)
      .def("frame",
           py::overload_cast<frame_t>(&Reconstruction::Frame),
           "frame_id"_a,
           "Direct accessor for a frame.",
           py::return_value_policy::reference_internal)
      .def_property_readonly("images",
                             &Reconstruction::Images,
                             py::return_value_policy::reference_internal)
      .def("image",
           py::overload_cast<image_t>(&Reconstruction::Image),
           "image_id"_a,
           "Direct accessor for an image.",
           py::return_value_policy::reference_internal)
      .def_property_readonly("points3D",
                             &Reconstruction::Points3D,
                             py::return_value_policy::reference_internal)
      .def("point3D",
           py::overload_cast<point3D_t>(&Reconstruction::Point3D),
           "point3D_id"_a,
           "Direct accessor for a Point3D.",
           py::return_value_policy::reference_internal)
      .def("reg_image_ids", &Reconstruction::RegImageIds)
      .def("reg_frame_ids", &Reconstruction::RegFrameIds)
      .def("point3D_ids", &Reconstruction::Point3DIds)
      .def("track_arrays",
           &TrackArrays,
           "Export points and tracks as contiguous NumPy arrays without "
           "creating Python objects per observation.")
      .def("add_points3D_from_arrays",
           &AddPoints3DFromArrays,
           "point_xyz"_a.noconvert(),
           "point_rgb"_a.noconvert(),
           "observation_offsets"_a.noconvert(),
           "observation_image_ids"_a.noconvert(),
           "observation_point2D_indices"_a.noconvert(),
           "Add 3D points and packed tracks without creating Python objects "
           "per observation. Returns the assigned point3D identifiers.")
      .def("exists_rig", &Reconstruction::ExistsRig, "rig_id"_a)
      .def("exists_camera", &Reconstruction::ExistsCamera, "camera_id"_a)
      .def("exists_frame", &Reconstruction::ExistsFrame, "frame_id"_a)
      .def("exists_image", &Reconstruction::ExistsImage, "image_id"_a)
      .def("exists_point3D", &Reconstruction::ExistsPoint3D, "point3D_id"_a)
      .def("is_valid",
           &Reconstruction::IsValid,
           "Check whether the reconstruction object is internally consistent.")
      .def("load", &Reconstruction::Load, "database_cache"_a)
      .def("tear_down", &Reconstruction::TearDown)
      .def("add_rig", &Reconstruction::AddRig, "rig"_a, "Add new rig.")
      .def("add_camera",
           &Reconstruction::AddCamera,
           "camera"_a,
           "Add new camera. There is only one camera per image, while multiple "
           "images might be taken by the same camera.")
      .def("add_camera_with_trivial_rig",
           &Reconstruction::AddCameraWithTrivialRig,
           "camera"_a,
           "Add a new camera and also create a trivial rig whose rig_id "
           "matches the "
           "camera_id. The camera becomes the rig's only sensor.")
      .def("add_frame", &Reconstruction::AddFrame, "frame"_a, "Add new frame.")
      .def(
          "add_image",
          &Reconstruction::AddImage,
          "image"_a,
          "Add new image. Its camera must have been added before. If its "
          "camera object is unset, it will be automatically populated from the "
          "added cameras.")
      .def("add_image_with_trivial_frame",
           py::overload_cast<Image>(&Reconstruction::AddImageWithTrivialFrame),
           "image"_a,
           "Add a new image and create a frame with the same ID (frame_id = "
           "image_id). "
           "Assumes a rig exists whose rig_id equals the camera_id of the "
           "image.")
      .def("add_image_with_trivial_frame",
           py::overload_cast<Image, const Rigid3d&>(
               &Reconstruction::AddImageWithTrivialFrame),
           "image"_a,
           "cam_from_world"_a,
           "Add a new image, create a trivial frame (frame_id = image_id), and "
           "also "
           "register the frame with an input pose.")

      .def("add_point3D",
           py::overload_cast<const Eigen::Vector3d&,
                             Track,
                             const Eigen::Vector3ub&>(
               &Reconstruction::AddPoint3D),
           "Add new 3D object, and return its unique ID.",
           "xyz"_a,
           "track"_a,
           "color"_a = Eigen::Vector3ub::Zero())
      .def("add_point3D_with_id",
           py::overload_cast<point3D_t, Point3D>(&Reconstruction::AddPoint3D),
           "point3D_id"_a,
           "point3D"_a,
           "Add new 3D point with known ID.")
      .def("add_observation",
           &Reconstruction::AddObservation,
           "point3D_id"_a,
           "track_element"_a,
           "Add observation to existing 3D point.")
      .def(
          "merge_points3D",
          &Reconstruction::MergePoints3D,
          "point3D_id1"_a,
          "point3D_id2"_a,
          "Merge two 3D points and return new identifier of new 3D point."
          "The location of the merged 3D point is a weighted average of the "
          "two original 3D point's locations according to their track lengths.")
      .def("delete_point3D",
           &Reconstruction::DeletePoint3D,
           "point3D_id"_a,
           "Delete a 3D point, and all its references in the observed images.")
      .def("delete_observation",
           &Reconstruction::DeleteObservation,
           "image_id"_a,
           "point2D_idx"_a,
           "Delete one observation from an image and the corresponding 3D "
           "point. Note that this deletes the entire 3D point, if the track "
           "has two elements prior to calling this method.")
      .def("delete_all_points2D_and_points3D",
           &Reconstruction::DeleteAllPoints2DAndPoints3D,
           "Delete all 2D points of all images and all 3D points.")
      .def("set_rigs_and_frames",
           &Reconstruction::SetRigsAndFrames,
           "rigs"_a,
           "frames"_a,
           "Set rigs and frames together.")
      .def("register_frame",
           &Reconstruction::RegisterFrame,
           "frame_id"_a,
           "Register an existing frame, and all its references.")
      .def("deregister_frame",
           &Reconstruction::DeRegisterFrame,
           "frame_id"_a,
           "De-register an existing frame, and all its references.")
      .def("normalize",
           &Reconstruction::Normalize,
           "fixed_scale"_a = false,
           "extent"_a = 10.0,
           "min_percentile"_a = 0.1,
           "max_percentile"_a = 0.9,
           "use_images"_a = true,
           "Normalize scene by scaling and translation to avoid degenerate "
           "visualization after bundle adjustment and to improve numerical "
           "stability of algorithms.\n\n"
           "Translates scene such that the mean of the camera centers or point "
           "locations are at the origin of the coordinate system.\n\n"
           "Scales scene such that the minimum and maximum camera centers "
           "(or points) are at the given `extent`, whereas `min_percentile` "
           "and `max_percentile` determine the minimum and maximum percentiles "
           "of the camera centers (or points) considered.")
      .def("transform",
           &Reconstruction::Transform,
           "new_from_old_world"_a,
           "Apply the 3D similarity transformation to all images and points.")
      .def("compute_centroid",
           &Reconstruction::ComputeCentroid,
           "min_percentile"_a = 0.0,
           "max_percentile"_a = 1.0,
           "use_images"_a = false)
      .def("compute_bounding_box",
           &Reconstruction::ComputeBoundingBox,
           "min_percentile"_a = 0.0,
           "max_percentile"_a = 1.0,
           "use_images"_a = false)
      .def("crop", &Reconstruction::Crop, "bbox"_a)
      .def("find_image_with_name",
           &Reconstruction::FindImageWithName,
           py::return_value_policy::reference_internal,
           "name"_a,
           "Find image with matching name. Returns None if no match is found.")
      .def("find_common_reg_image_ids",
           &Reconstruction::FindCommonRegImageIds,
           "other"_a,
           "Find images that are both present in this and the given "
           "reconstruction.")
      .def("transcribe_image_ids_to_database",
           &Reconstruction::TranscribeImageIdsToDatabase,
           "database"_a,
           "Update image identifiers to match the database by name.")
      .def("update_point_3d_errors", &Reconstruction::UpdatePoint3DErrors)
      .def("compute_num_observations", &Reconstruction::ComputeNumObservations)
      .def("compute_mean_track_length", &Reconstruction::ComputeMeanTrackLength)
      .def("compute_mean_observations_per_reg_image",
           &Reconstruction::ComputeMeanObservationsPerRegImage)
      .def("compute_mean_reprojection_error",
           &Reconstruction::ComputeMeanReprojectionError)
      .def("import_PLY",
           py::overload_cast<const std::filesystem::path&>(
               &Reconstruction::ImportPLY),
           "path"_a,
           "Import from PLY format. Note that these import functions are "
           "only intended for visualization of data and not usable for "
           "reconstruction.")
      .def("export_PLY",
           &ExportPLY,
           "output_path"_a,
           "Export 3D points to PLY format (.ply).")
      .def("extract_colors_for_image",
           &Reconstruction::ExtractColorsForImage,
           "image_id"_a,
           "path"_a,
           "Extract colors for 3D points of given image. Colors will be "
           "extracted only for 3D points which are completely black. "
           "Return True if the image could be read at the given path.")
      .def("extract_colors_for_all_images",
           &Reconstruction::ExtractColorsForAllImages,
           "Extract colors for all 3D points by computing the mean color of "
           "all images.",
           "path"_a,
           "num_threads"_a = -1)
      .def("create_image_dirs",
           &Reconstruction::CreateImageDirs,
           "path"_a,
           "Create all image sub-directories in the given path.")
      .def("__copy__",
           [](const Reconstruction& self) { return Reconstruction(self); })
      .def("__deepcopy__",
           [](const Reconstruction& self, const py::dict&) {
             return Reconstruction(self);
           })
      .def("__repr__", &CreateRepresentation<Reconstruction>)
      .def("summary", [](const Reconstruction& self) {
        std::ostringstream ss;
        ss << "Reconstruction:"
           << "\n\tnum_rigs = " << self.NumRigs()
           << "\n\tnum_cameras = " << self.NumCameras()
           << "\n\tnum_frames = " << self.NumFrames()
           << "\n\tnum_reg_frames = " << self.NumRegFrames()
           << "\n\tnum_images = " << self.NumImages()
           << "\n\tnum_points3D = " << self.NumPoints3D()
           << "\n\tnum_observations = " << self.ComputeNumObservations()
           << "\n\tmean_track_length = " << self.ComputeMeanTrackLength()
           << "\n\tmean_observations_per_image = "
           << self.ComputeMeanObservationsPerRegImage()
           << "\n\tmean_reprojection_error = "
           << self.ComputeMeanReprojectionError();
        return ss.str();
      });
}
