#include "dynamic_store.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<DynamicStore>(m, "DynamicStore").def(py::init<>());
}