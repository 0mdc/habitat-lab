// Copyright (c) Meta Platforms, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// Copied from: https://github.com/shacklettbp/bps-nav/blob/master/simulator/pytorch.cpp

#include <torch/extension.h>

#include <array>
#include <cstdint>

namespace py = pybind11;

// Sanity check for torch extension compilation.
//
bool loaded() {
    return true;
}

// Returns whether the pointer contained within a capsule is valid.
//
bool isCapsuleValid(const py::capsule& capsule) {
  return capsule.get_pointer();
}

// Create a tensor that references CUDA memory.
//
at::Tensor convertToTensorColor(const py::capsule& ptr_capsule,
                                int dev_id,
                                uint32_t batch_size,
                                const std::array<uint32_t, 2>& resolution) {
  uint8_t* dev_ptr(ptr_capsule);

  std::array<int64_t, 4> sizes{{batch_size, resolution[0], resolution[1], 4}};

  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .device(torch::kCUDA, (short)dev_id);

  return torch::from_blob(dev_ptr, sizes, options);
}

// Create a tensor that references CUDA memory.
//
at::Tensor convertToTensorDepth(const py::capsule& ptr_capsule,
                                int dev_id,
                                uint32_t batch_size,
                                const std::array<uint32_t, 2>& resolution) {
  float* dev_ptr(ptr_capsule);

  std::array<int64_t, 3> sizes{{batch_size, resolution[0], resolution[1]}};

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .device(torch::kCUDA, (short)dev_id);

  return torch::from_blob(dev_ptr, sizes, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("loaded", &loaded);
  m.def("is_capsule_valid", &isCapsuleValid);
  m.def("make_color_tensor", &convertToTensorColor);
  m.def("make_depth_tensor", &convertToTensorDepth);
}
