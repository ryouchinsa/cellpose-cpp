cmake_minimum_required(VERSION 3.10)
project(cyto3_cpp)

find_package(Torch REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)

message("TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

add_library(cyto3_cpp SHARED cyto3.h cyto3.cpp fill_voids.h fill_voids.cpp)

target_compile_features(cyto3_cpp PUBLIC cxx_std_14)

target_include_directories(
  cyto3_cpp SYSTEM PUBLIC 
  ${ONNXRUNTIME_ROOT_DIR}/include
)

if (APPLE)
  set(onnxruntime_lib ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.dylib)
else()
  set(onnxruntime_lib ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)
endif()
target_link_libraries(
  cyto3_cpp PUBLIC
  ${onnxruntime_lib}
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  gflags
)

add_executable(cyto3_cpp_test test.cpp)
target_link_libraries(
  cyto3_cpp_test PRIVATE
  cyto3_cpp
)

