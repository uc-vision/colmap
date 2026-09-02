#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"

#include "pycolmap/pybind11_extension.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include <Eigen/Core>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style>;
using Int32Array = py::array_t<int32_t, py::array::c_style>;
using Int64Array = py::array_t<int64_t, py::array::c_style>;
using Uint32Array = py::array_t<uint32_t, py::array::c_style>;
using Uint64Array = py::array_t<uint64_t, py::array::c_style>;

constexpr float kObservationTolerancePixels = 1.0f;
constexpr float kObservationToleranceSquared =
    kObservationTolerancePixels * kObservationTolerancePixels;
constexpr float kObservationCellScale = 1.0f / kObservationTolerancePixels;
constexpr uint32_t kInvalidIndex = std::numeric_limits<uint32_t>::max();

struct RowTrackSource {
  RowTrackSource(Int64Array observation_offsets,
                 Int32Array observation_image_indices,
                 FloatArray observation_xy,
                 Int32Array image_to_shared_index)
      : observation_offsets(std::move(observation_offsets)),
        observation_image_indices(std::move(observation_image_indices)),
        observation_xy(std::move(observation_xy)),
        image_to_shared_index(std::move(image_to_shared_index)) {}

  Int64Array observation_offsets;
  Int32Array observation_image_indices;
  FloatArray observation_xy;
  Int32Array image_to_shared_index;
};

struct RowTrackJoin {
  py::array_t<uint32_t> row_point_index;
  uint32_t point_count;
  py::array_t<uint32_t> point_track_offsets;
  py::array_t<uint32_t> point_track_indices;
  py::array_t<uint32_t> point_observation_counts;
  py::array_t<int64_t> duplicate_observation_offsets;
  py::array_t<uint32_t> duplicate_observation_indices;
};

struct FeatureTracks {
  Uint32Array observation_offsets;
  Uint64Array observation_codes;
};

struct FeatureEdge {
  uint64_t first;
  uint64_t second;

  bool operator<(const FeatureEdge& other) const {
    return first < other.first ||
           (first == other.first && second < other.second);
  }

  bool operator==(const FeatureEdge& other) const {
    return first == other.first && second == other.second;
  }
};

struct CanonicalObservation {
  Eigen::Vector2f xy;
  uint32_t track;
  uint32_t next_in_cell;
};

struct SharedImageObservations {
  std::vector<CanonicalObservation> observations;
  FlatHashMap<uint64_t, uint32_t> cell_heads;
};

struct PendingObservation {
  uint32_t shared_image;
  Eigen::Vector2f xy;
  uint32_t track;
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

Eigen::Map<const Eigen::Vector2f> ObservationXY(const RowTrackSource& source,
                                                uint32_t observation) {
  return Eigen::Map<const Eigen::Vector2f>(source.observation_xy.data() +
                                           2 * observation);
}

Eigen::Vector2i ObservationCell(const Eigen::Vector2f& xy) {
  return (xy * kObservationCellScale).array().floor().cast<int>().matrix();
}

uint64_t CellKey(const Eigen::Vector2i& cell) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(cell.x())) << 32) |
         static_cast<uint32_t>(cell.y());
}

void AddCanonicalObservation(const Eigen::Vector2f& xy,
                             uint32_t track,
                             SharedImageObservations* image) {
  const uint64_t cell_key = CellKey(ObservationCell(xy));
  const auto [cell_it, inserted] =
      image->cell_heads.try_emplace(cell_key, kInvalidIndex);
  const uint32_t observation =
      static_cast<uint32_t>(image->observations.size());
  image->observations.push_back(CanonicalObservation{
      xy, track, inserted ? kInvalidIndex : cell_it->second});
  cell_it->second = observation;
}

uint32_t FindCanonicalObservation(const SharedImageObservations& image,
                                  const Eigen::Vector2f& xy) {
  const Eigen::Vector2i cell = ObservationCell(xy);
  uint32_t nearest_observation = kInvalidIndex;
  float nearest_distance = std::numeric_limits<float>::infinity();
  for (int cell_y = -1; cell_y <= 1; ++cell_y) {
    for (int cell_x = -1; cell_x <= 1; ++cell_x) {
      const auto cell_it = image.cell_heads.find(
          CellKey(cell + Eigen::Vector2i(cell_x, cell_y)));
      if (cell_it == image.cell_heads.end()) {
        continue;
      }
      for (uint32_t observation = cell_it->second; observation != kInvalidIndex;
           observation = image.observations[observation].next_in_cell) {
        const float squared_distance =
            (image.observations[observation].xy - xy).squaredNorm();
        if (squared_distance <= kObservationToleranceSquared &&
            (squared_distance < nearest_distance ||
             (squared_distance == nearest_distance &&
              observation < nearest_observation))) {
          nearest_observation = observation;
          nearest_distance = squared_distance;
        }
      }
    }
  }
  return nearest_observation;
}

size_t NumSharedImages(const std::vector<const RowTrackSource*>& sources) {
  size_t num_shared_images = 0;
  for (const RowTrackSource* source : sources) {
    for (ssize_t image = 0; image < source->image_to_shared_index.shape(0);
         ++image) {
      const int32_t shared_image = source->image_to_shared_index.data()[image];
      if (shared_image >= 0) {
        num_shared_images =
            std::max(num_shared_images, static_cast<size_t>(shared_image + 1));
      }
    }
  }
  return num_shared_images;
}

RowTrackJoin JoinRowTracks(const std::vector<const RowTrackSource*>& sources) {
  size_t num_tracks = 0;
  for (const RowTrackSource* source : sources) {
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

    std::vector<SharedImageObservations> shared_images(
        NumSharedImages(sources));
    uint32_t track_base = 0;
    for (const RowTrackSource* source_pointer : sources) {
      const RowTrackSource& source = *source_pointer;
      const int64_t* observation_offsets = source.observation_offsets.data();
      const int32_t* observation_image_indices =
          source.observation_image_indices.data();
      const int32_t* image_to_shared_index =
          source.image_to_shared_index.data();
      const uint32_t source_track_count =
          source.observation_offsets.shape(0) - 1;
      std::vector<PendingObservation> pending_observations;
      for (uint32_t track = 0; track < source_track_count; ++track) {
        const uint32_t global_track = track_base + track;
        track_observation_counts[global_track] = static_cast<uint32_t>(
            observation_offsets[track + 1] - observation_offsets[track]);
        for (int64_t observation = observation_offsets[track];
             observation < observation_offsets[track + 1];
             ++observation) {
          const int32_t shared_image =
              image_to_shared_index[observation_image_indices[observation]];
          if (shared_image >= 0) {
            const Eigen::Vector2f xy =
                ObservationXY(source, static_cast<uint32_t>(observation));
            SharedImageObservations& image = shared_images[shared_image];
            const uint32_t canonical_observation =
                FindCanonicalObservation(image, xy);
            if (canonical_observation == kInvalidIndex) {
              pending_observations.push_back(PendingObservation{
                  static_cast<uint32_t>(shared_image), xy, global_track});
            } else {
              duplicate_observation_indices.push_back(
                  static_cast<uint32_t>(observation));
              --track_observation_counts[global_track];
              UnionRoots(parent,
                         global_track,
                         image.observations[canonical_observation].track);
            }
          }
        }
      }
      for (const PendingObservation& observation : pending_observations) {
        AddCanonicalObservation(observation.xy,
                                observation.track,
                                &shared_images[observation.shared_image]);
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
  return RowTrackJoin{std::move(row_point_index),
                      point_count,
                      std::move(point_track_offset_array),
                      std::move(point_track_index_array),
                      std::move(point_observation_count_array),
                      std::move(duplicate_offsets),
                      std::move(duplicate_indices)};
}

FeatureTracks BuildFeatureTracks(const Uint32Array& image_ids1,
                                 const Uint32Array& image_ids2,
                                 const Uint64Array& match_offsets,
                                 const Uint32Array& matches) {
  THROW_CHECK_EQ(image_ids1.ndim(), 1);
  THROW_CHECK_EQ(image_ids2.ndim(), 1);
  THROW_CHECK_EQ(image_ids1.shape(0), image_ids2.shape(0));
  const size_t num_pairs = image_ids1.shape(0);
  THROW_CHECK_EQ(match_offsets.ndim(), 1);
  THROW_CHECK_EQ(match_offsets.shape(0), num_pairs + 1);
  THROW_CHECK_EQ(matches.ndim(), 2);
  THROW_CHECK_EQ(matches.shape(1), 2);
  THROW_CHECK_EQ(match_offsets.data()[0], 0);
  THROW_CHECK_EQ(match_offsets.data()[num_pairs], matches.shape(0));

  std::vector<uint32_t> output_offsets;
  std::vector<uint64_t> output_codes;
  {
    py::gil_scoped_release release;
    std::vector<FeatureEdge> edges;
    edges.reserve(matches.shape(0));
    for (size_t pair = 0; pair < num_pairs; ++pair) {
      const uint64_t image1 = image_ids1.data()[pair];
      const uint64_t image2 = image_ids2.data()[pair];
      for (uint64_t match = match_offsets.data()[pair];
           match < match_offsets.data()[pair + 1];
           ++match) {
        uint64_t first = (image1 << 32) | matches.data()[2 * match];
        uint64_t second = (image2 << 32) | matches.data()[2 * match + 1];
        if (second < first) {
          std::swap(first, second);
        }
        edges.push_back({first, second});
      }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    std::vector<uint32_t> image_ids;
    image_ids.reserve(2 * num_pairs);
    image_ids.insert(
        image_ids.end(), image_ids1.data(), image_ids1.data() + num_pairs);
    image_ids.insert(
        image_ids.end(), image_ids2.data(), image_ids2.data() + num_pairs);
    std::sort(image_ids.begin(), image_ids.end());
    image_ids.erase(std::unique(image_ids.begin(), image_ids.end()),
                    image_ids.end());
    FlatHashMap<uint32_t, uint32_t> image_slots;
    for (uint32_t slot = 0; slot < image_ids.size(); ++slot) {
      image_slots.emplace(image_ids[slot], slot);
    }
    const size_t image_mask_words = (image_ids.size() + 63) / 64;

    FlatHashMap<uint64_t, uint32_t> nodes;
    nodes.reserve(edges.size());
    std::vector<uint32_t> parent;
    std::vector<uint64_t> codes;
    std::vector<uint64_t> image_masks;
    auto add_node = [&](const uint64_t code) {
      const uint32_t node = parent.size();
      nodes.emplace(code, node);
      parent.push_back(node);
      codes.push_back(code);
      image_masks.resize(image_masks.size() + image_mask_words, 0);
      const uint32_t image_slot = image_slots.at(code >> 32);
      image_masks[node * image_mask_words + image_slot / 64] |=
          uint64_t{1} << (image_slot % 64);
      return node;
    };
    auto node_for = [&](const uint64_t code) {
      const auto node = nodes.find(code);
      return node == nodes.end() ? add_node(code) : node->second;
    };
    auto find_root = [&parent](uint32_t node) {
      while (parent[node] != node) {
        parent[node] = parent[parent[node]];
        node = parent[node];
      }
      return node;
    };

    for (const FeatureEdge& edge : edges) {
      uint32_t root1 = find_root(node_for(edge.first));
      uint32_t root2 = find_root(node_for(edge.second));
      if (root1 == root2) {
        continue;
      }
      bool images_overlap = false;
      for (size_t word = 0; word < image_mask_words; ++word) {
        images_overlap |= image_masks[root1 * image_mask_words + word] &
                          image_masks[root2 * image_mask_words + word];
      }
      if (images_overlap) {
        continue;
      }
      if (codes[root2] < codes[root1]) {
        std::swap(root1, root2);
      }
      parent[root2] = root1;
      for (size_t word = 0; word < image_mask_words; ++word) {
        image_masks[root1 * image_mask_words + word] |=
            image_masks[root2 * image_mask_words + word];
      }
    }

    std::vector<uint32_t> component_sizes(parent.size(), 0);
    for (uint32_t node = 0; node < parent.size(); ++node) {
      parent[node] = find_root(node);
      ++component_sizes[parent[node]];
    }
    std::vector<uint32_t> roots;
    for (uint32_t node = 0; node < parent.size(); ++node) {
      if (parent[node] == node && component_sizes[node] >= 2) {
        roots.push_back(node);
      }
    }
    std::sort(
        roots.begin(), roots.end(), [&codes](uint32_t first, uint32_t second) {
          return codes[first] < codes[second];
        });
    std::vector<uint32_t> track_indices(parent.size(), kInvalidIndex);
    output_offsets.resize(roots.size() + 1, 0);
    for (size_t track = 0; track < roots.size(); ++track) {
      track_indices[roots[track]] = track;
      output_offsets[track + 1] =
          output_offsets[track] + component_sizes[roots[track]];
    }
    output_codes.resize(output_offsets.back());
    std::vector<uint32_t> cursors = output_offsets;
    for (uint32_t node = 0; node < parent.size(); ++node) {
      const uint32_t track = track_indices[parent[node]];
      if (track != kInvalidIndex) {
        output_codes[cursors[track]++] = codes[node];
      }
    }
    for (size_t track = 0; track < roots.size(); ++track) {
      std::sort(output_codes.begin() + output_offsets[track],
                output_codes.begin() + output_offsets[track + 1]);
    }
  }

  Uint32Array observation_offsets(output_offsets.size());
  std::copy(output_offsets.begin(),
            output_offsets.end(),
            observation_offsets.mutable_data());
  Uint64Array observation_codes(output_codes.size());
  std::copy(output_codes.begin(),
            output_codes.end(),
            observation_codes.mutable_data());
  return {std::move(observation_offsets), std::move(observation_codes)};
}

}  // namespace

void BindRowTrackJoin(py::module& m) {
  py::classh<RowTrackSource>(m, "RowTrackSource")
      .def(py::init<Int64Array, Int32Array, FloatArray, Int32Array>(),
           "observation_offsets"_a,
           "observation_image_indices"_a,
           "observation_xy"_a,
           "image_to_shared_index"_a)
      .def_readonly("observation_offsets", &RowTrackSource::observation_offsets)
      .def_readonly("observation_image_indices",
                    &RowTrackSource::observation_image_indices)
      .def_readonly("observation_xy", &RowTrackSource::observation_xy)
      .def_readonly("image_to_shared_index",
                    &RowTrackSource::image_to_shared_index);

  py::classh<RowTrackJoin>(m, "RowTrackJoin")
      .def_readonly("row_point_index", &RowTrackJoin::row_point_index)
      .def_readonly("point_count", &RowTrackJoin::point_count)
      .def_readonly("point_track_offsets", &RowTrackJoin::point_track_offsets)
      .def_readonly("point_track_indices", &RowTrackJoin::point_track_indices)
      .def_readonly("point_observation_counts",
                    &RowTrackJoin::point_observation_counts)
      .def_readonly("duplicate_observation_offsets",
                    &RowTrackJoin::duplicate_observation_offsets)
      .def_readonly("duplicate_observation_indices",
                    &RowTrackJoin::duplicate_observation_indices);

  py::classh<FeatureTracks>(m, "FeatureTracks")
      .def_readonly("observation_offsets", &FeatureTracks::observation_offsets)
      .def_readonly("observation_codes", &FeatureTracks::observation_codes);

  m.def("join_row_tracks",
        &JoinRowTracks,
        "sources"_a,
        "Join row tracks by shared-image observation coordinates.");
  m.def(
      "build_feature_tracks",
      &BuildFeatureTracks,
      "image_ids1"_a,
      "image_ids2"_a,
      "match_offsets"_a,
      "matches"_a,
      "Build deterministic feature tracks with at most one feature per image.");
}
