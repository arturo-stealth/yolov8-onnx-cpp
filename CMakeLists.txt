cmake_minimum_required(VERSION 3.0.0)
project(yolov8_onnxruntime)
add_compile_options(-std=c++17)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(ONNXRUNTIME_DIR /opt/onnxruntime)  # onnxruntime root

# Find OpenVINO
find_package(OpenVINO REQUIRED)

# Ensure ONNXRUNTIME_DIR exists
if(NOT EXISTS ${ONNXRUNTIME_DIR})
  message(FATAL_ERROR "ONNXRUNTIME_DIR not found")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_DIR}/include
)

set(${PROJECT_NAME}_CPP_SOURCES
  src/nn/autobackend.cpp
  src/nn/onnx_model_base.cpp 
  src/utils/augment.cpp
  src/utils/common.cpp
  src/utils/ops.cpp
)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_CPP_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)

add_executable(${PROJECT_NAME}_test src/main.cpp)
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME} ${OpenCV_LIBS})

# Install targets
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_test
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# Package configuration
if(NOT TARGET CPack)
  include(CPack)
endif()

set(CPACK_PACKAGE_NAME "yolov8_onnxruntime")
set(CPACK_PACKAGE_VENDOR "YourCompany")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "YOLOv8 ONNX Runtime Library")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "yolov8_onnxruntime")

set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Your Name <your.email@example.com>")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libopencv-dev, libonnxruntime-dev")
set(CPACK_RPM_PACKAGE_REQUIRES "opencv, onnxruntime")

include(CPack)

# CMake package config
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/yolov8_onnxruntimeConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/yolov8_onnxruntimeConfig.cmake @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/yolov8_onnxruntimeConfig.cmake
  DESTINATION lib/cmake/yolov8_onnxruntime
)

# Find package config
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/yolov8_onnxruntimeConfigVersion.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/yolov8_onnxruntimeConfigVersion.cmake @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/yolov8_onnxruntimeConfigVersion.cmake
  DESTINATION lib/cmake/yolov8_onnxruntime
)
