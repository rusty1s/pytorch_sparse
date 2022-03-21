cmake_minimum_required(VERSION 3.12)
project(torchsparse)
set(CMAKE_CXX_STANDARD 14)
set(TORCHSPARSE_VERSION 0.6.13)

option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_PYTHON "Link to Python when building" OFF)

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

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

list(APPEND ALLOW_LISTED csrc csrc/cpu)
if(WITH_CUDA)
  file(APPEND ALLOW_LISTED csrc/cuda)
endif()

FOREACH(DIR ${ALLOW_LISTED})
  file(GLOB ALL_SOURCES ${ALL_SOURCES}, ${DIR}/*.*)
ENDFOREACH()

add_library(${PROJECT_NAME} SHARED ${ALL_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

if (WITH_PYTHON)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
  EXPORT_NAME TorchSparse
  INSTALL_RPATH ${TORCH_INSTALL_PREFIX}/lib)

include_directories(csrc)

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
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT TorchSparseTargets
  NAMESPACE TorchSparse::
  DESTINATION ${TORCHSPARSE_CMAKECONFIG_INSTALL_DIR})


FOREACH(INPUT_DIR ${ALLOW_LISTED})
    string(REPLACE "csrc" "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}" OUTPUT_DIR ${INPUT_DIR})
    file(GLOB INPUT_FILES ${INPUT_DIR}/*.*)
    install(FILES ${INPUT_FILES} DESTINATION ${OUTPUT_DIR})
ENDFOREACH()
