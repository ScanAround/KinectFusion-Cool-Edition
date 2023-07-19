cmake_minimum_required(VERSION 3.10)
project(KinectFusion)

set(CMAKE_CXX_STANDARD 20)  # Use C++20
set(CMAKE_BUILD_TYPE Debug)  # Set build type to Debug which will automatically add -g flag

# In case you are using any additional library add them here
# If you are using Eigen, assuming it is installed in a standard location, uncomment these lines:
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(/usr/include)  # Path to the FreeImage headers

add_library(KinectFusionLib
  voxel_grid.cpp
  voxel.cpp
  kinect_fusion_utility.cpp
)

add_executable(KinectFusionApp test_tsdf.cpp)

# Linking your KinectFusion library to your executable
target_link_libraries(KinectFusionApp KinectFusionLib ${OpenCV_LIBS} /usr/lib/x86_64-linux-gnu/libfreeimage.so)  # Path to the FreeImage library

# Suppress deprecated warnings and some others
# Narrowing warnings should be solved ideally
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(KinectFusionLib PRIVATE -Wno-deprecated -Wno-ignored-attributes -Wno-narrowing)
  target_compile_options(KinectFusionApp PRIVATE -Wno-deprecated -Wno-ignored-attributes -Wno-narrowing)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(KinectFusionLib PRIVATE -Wno-deprecated-enum-enum-conversion -Wno-ignored-attributes -Wno-c++11-narrowing)
  target_compile_options(KinectFusionApp PRIVATE -Wno-deprecated-enum-enum-conversion -Wno-ignored-attributes -Wno-c++11-narrowing)
endif()

# This can be run as follows:
# mkdir build
# cd build
# cmake ..
# make
# ./KinectFusionApp