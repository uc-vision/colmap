#include "colmap/util/hash_containers.h"

#include "pycolmap/pybind11_extension.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

using Int32Array = py::array_t<int32_t, py::array::c_style>;
using Int64Array = py::array_t<int64_t, py::array::c_style>;

struct RowTrackIdentitySource {
  RowTrackIdentitySource(Int64Array observation_offsets,
                         Int32Array observation_image_indices,
                         Int64Array observation_feature_indices,
                         Int32Array image_to_shared_index)
      : observation_offsets(std::move(observation_offsets)),
        observation_image_indices(std::move(observation_image_indices)),
        observation_feature_indices(std::move(observation_feature_indices)),
        image_to_shared_index(std::move(image_to_shared_index)) {}

  Int64Array observation_offsets;
  Int32Array observation_image_indices;
  Int64Array observation_feature_indices;
  Int32Array image_to_shared_index;
};

struct RowTrackIdentityJoin {
  py::array_t<uint32_t> row_point_index;
  uint32_t point_count;
  py::array_t<uint32_t> point_track_offsets;
  py::array_t<uint32_t> point_track_indices;
  py::array_t<uint32_t> point_observation_counts;
  py::array_t<int64_t> duplicate_observation_offsets;
  py::array_t<uint32_t> duplicate_observation_indices;
};

uint32_t FindRoot(uint32_t* parent, uint32_t node) {
  while (parent[node] != node) {
    parent[node] = parent[parent[node]];
    node = parent[node];
  }
  return node;
}

void UnionRoots(uint32_t* parent, uint32_t first, uint32_t second) {
  first = FindRoot(parent, first);
  second = FindRoot(parent, second);
  if (first != second) {
    parent[std::max(first, second)] = std::min(first, second);
  }
}

RowTrackIdentityJoin JoinRowTrackIdentities(
    const std::vector<const RowTrackIdentitySource*>& sources) {
  size_t num_tracks = 0;
  for (const RowTrackIdentitySource* source : sources) {
    num_tracks += source->observation_offsets.shape(0) - 1;
  }

  py::array_t<uint32_t> row_point_index(num_tracks);
  uint32_t* parent = row_point_index.mutable_data();
  std::vector<int64_t> duplicate_observation_offsets{0};
  std::vector<uint32_t> duplicate_observation_indices;
  std::vector<uint32_t> track_observation_counts(num_tracks);
  std::vector<uint32_t> point_track_offsets;
  std::vector<uint32_t> point_track_indices;
  std::vector<uint32_t> point_observation_counts;
  uint32_t point_count = 0;
  {
    py::gil_scoped_release release;
    std::iota(parent, parent + num_tracks, uint32_t{0});

    FlatHashMap<uint64_t, uint32_t> track_by_identity;
    track_by_identity.reserve(num_tracks);
    uint32_t track_base = 0;
    for (const RowTrackIdentitySource* source : sources) {
      const int64_t* observation_offsets = source->observation_offsets.data();
      const int32_t* observation_image_indices =
          source->observation_image_indices.data();
      const int64_t* observation_feature_indices =
          source->observation_feature_indices.data();
      const int32_t* image_to_shared_index =
          source->image_to_shared_index.data();
      const uint32_t source_track_count =
          source->observation_offsets.shape(0) - 1;
      for (uint32_t track = 0; track < source_track_count; ++track) {
        const uint32_t global_track = track_base + track;
        track_observation_counts[global_track] = static_cast<uint32_t>(
            observation_offsets[track + 1] - observation_offsets[track]);
        for (int64_t observation = observation_offsets[track];
             observation < observation_offsets[track + 1];
             ++observation) {
          const int32_t shared_image =
              image_to_shared_index[observation_image_indices[observation]];
          if (shared_image < 0) {
            continue;
          }
          const uint64_t identity =
              (static_cast<uint64_t>(shared_image) << 32) |
              static_cast<uint32_t>(observation_feature_indices[observation]);
          const auto [identity_it, inserted] =
              track_by_identity.try_emplace(identity, global_track);
          if (!inserted) {
            duplicate_observation_indices.push_back(
                static_cast<uint32_t>(observation));
            --track_observation_counts[global_track];
            UnionRoots(parent, global_track, identity_it->second);
          }
        }
      }
      track_base += source_track_count;
      duplicate_observation_offsets.push_back(
          static_cast<int64_t>(duplicate_observation_indices.size()));
    }

    for (uint32_t track = 0; track < num_tracks; ++track) {
      parent[track] = FindRoot(parent, track);
    }
    for (uint32_t track = 0; track < num_tracks; ++track) {
      const uint32_t root = parent[track];
      parent[track] = root == track ? point_count++ : parent[root];
    }

    point_track_offsets.assign(point_count + 1, 0);
    point_observation_counts.assign(point_count, 0);
    for (uint32_t track = 0; track < num_tracks; ++track) {
      const uint32_t point = parent[track];
      ++point_track_offsets[point + 1];
      point_observation_counts[point] += track_observation_counts[track];
    }
    std::partial_sum(point_track_offsets.begin(),
                     point_track_offsets.end(),
                     point_track_offsets.begin());
    point_track_indices.resize(num_tracks);
    std::vector<uint32_t> point_track_cursor = point_track_offsets;
    for (uint32_t track = 0; track < num_tracks; ++track) {
      point_track_indices[point_track_cursor[parent[track]]++] = track;
    }
  }

  py::array_t<uint32_t> point_track_offset_array(point_track_offsets.size());
  std::copy(point_track_offsets.begin(),
            point_track_offsets.end(),
            point_track_offset_array.mutable_data());
  py::array_t<uint32_t> point_track_index_array(point_track_indices.size());
  std::copy(point_track_indices.begin(),
            point_track_indices.end(),
            point_track_index_array.mutable_data());
  py::array_t<uint32_t> point_observation_count_array(
      point_observation_counts.size());
  std::copy(point_observation_counts.begin(),
            point_observation_counts.end(),
            point_observation_count_array.mutable_data());
  py::array_t<int64_t> duplicate_offsets(duplicate_observation_offsets.size());
  std::copy(duplicate_observation_offsets.begin(),
            duplicate_observation_offsets.end(),
            duplicate_offsets.mutable_data());
  py::array_t<uint32_t> duplicate_indices(duplicate_observation_indices.size());
  std::copy(duplicate_observation_indices.begin(),
            duplicate_observation_indices.end(),
            duplicate_indices.mutable_data());
  return RowTrackIdentityJoin{std::move(row_point_index),
                              point_count,
                              std::move(point_track_offset_array),
                              std::move(point_track_index_array),
                              std::move(point_observation_count_array),
                              std::move(duplicate_offsets),
                              std::move(duplicate_indices)};
}

}  // namespace

void BindRowTrackJoin(py::module& m) {
  py::classh<RowTrackIdentitySource>(m, "RowTrackIdentitySource")
      .def(py::init<Int64Array, Int32Array, Int64Array, Int32Array>(),
           "observation_offsets"_a,
           "observation_image_indices"_a,
           "observation_feature_indices"_a,
           "image_to_shared_index"_a)
      .def_readonly("observation_offsets",
                    &RowTrackIdentitySource::observation_offsets)
      .def_readonly("observation_image_indices",
                    &RowTrackIdentitySource::observation_image_indices)
      .def_readonly("observation_feature_indices",
                    &RowTrackIdentitySource::observation_feature_indices)
      .def_readonly("image_to_shared_index",
                    &RowTrackIdentitySource::image_to_shared_index);

  py::classh<RowTrackIdentityJoin>(m, "RowTrackIdentityJoin")
      .def_readonly("row_point_index", &RowTrackIdentityJoin::row_point_index)
      .def_readonly("point_count", &RowTrackIdentityJoin::point_count)
      .def_readonly("point_track_offsets",
                    &RowTrackIdentityJoin::point_track_offsets)
      .def_readonly("point_track_indices",
                    &RowTrackIdentityJoin::point_track_indices)
      .def_readonly("point_observation_counts",
                    &RowTrackIdentityJoin::point_observation_counts)
      .def_readonly("duplicate_observation_offsets",
                    &RowTrackIdentityJoin::duplicate_observation_offsets)
      .def_readonly("duplicate_observation_indices",
                    &RowTrackIdentityJoin::duplicate_observation_indices);

  m.def("join_row_track_identities",
        &JoinRowTrackIdentities,
        "sources"_a,
        "Join exact image-feature track identities into row point indices.");
}
