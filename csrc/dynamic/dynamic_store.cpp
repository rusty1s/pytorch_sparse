#include "dynamic_store.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<DynamicStore<float>>(m, "DynamicStore")
      .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t>())
      .def("append_block", &DynamicStore<float>::append_block);
  py::class_<DynamicStore<int>>(m, "DynamicIntStore")
      .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t>())
      .def("append_block", &DynamicStore<int>::append_block);
}