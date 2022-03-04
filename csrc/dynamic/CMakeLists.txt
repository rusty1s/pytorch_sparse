cmake_minimum_required(VERSION 3.12)
project(torchsparse_dynamic)
set(CMAKE_CXX_STANDARD 17)

file(GLOB HEADERS *.h)
file(GLOB SOURCES *.cpp)

find_package(Python3 COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC .)

target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
