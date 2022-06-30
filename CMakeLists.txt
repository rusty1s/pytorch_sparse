cmake_minimum_required(VERSION 3.0)
project(torchsparse)
set(CMAKE_CXX_STANDARD 14)
set(TORCHSPARSE_VERSION 0.6.14)

option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_PYTHON "Link to Python when building" ON)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

if (WITH_PYTHON)
  add_definitions(-DWITH_PYTHON)
  find_package(Python3 COMPONENTS Development)
endif()
find_package(Torch REQUIRED)

file(GLOB HEADERS csrc/*.h)
file(GLOB OPERATOR_SOURCES csrc/*.* csrc/cpu/*.*)
if(WITH_CUDA)
  file(GLOB OPERATOR_SOURCES ${OPERATOR_SOURCES} csrc/cuda/*.h csrc/cuda/*.cu)
endif()

add_library(${PROJECT_NAME} SHARED ${OPERATOR_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
if (WITH_PYTHON)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME TorchSparse)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${HEADERS}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TORCHSPARSE_CMAKECONFIG_INSTALL_DIR "share/cmake/TorchSparse" CACHE STRING "install path for TorchSparseConfig.cmake")

configure_package_config_file(cmake/TorchSparseConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/TorchSparseConfig.cmake"
  INSTALL_DESTINATION ${TORCHSPARSE_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/TorchSparseConfigVersion.cmake
	VERSION ${TORCHSPARSE_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TorchSparseConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/TorchSparseConfigVersion.cmake
  DESTINATION ${TORCHSPARSE_CMAKECONFIG_INSTALL_DIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT TorchSparseTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(EXPORT TorchSparseTargets
  NAMESPACE TorchSparse::
  DESTINATION ${TORCHSPARSE_CMAKECONFIG_INSTALL_DIR})

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
install(FILES
  csrc/cpu/convert_cpu.h
  csrc/cpu/diag_cpu.h
  csrc/cpu/metis_cpu.h
  csrc/cpu/rw_cpu.h
  csrc/cpu/saint_cpu.h
  csrc/cpu/sample_cpu.h
  csrc/cpu/spmm_cpu.h
  csrc/cpu/spspmm_cpu.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cpu)
if(WITH_CUDA)
  install(FILES
    csrc/cuda/convert_cuda.h
    csrc/cuda/diag_cuda.h
    csrc/cuda/rw_cuda.h
    csrc/cuda/spmm_cuda.h
    csrc/cuda/spspmm_cuda.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cuda)
endif()

if(WITH_CUDA)
  set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
  set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
endif()
