cmake_minimum_required(VERSION 3.10)
project(KinectFusion)

set(CMAKE_CXX_STANDARD 20)  # Use C++20

set(CMAKE_BUILD_TYPE Debug)  # Set build type to Debug which will automatically add -g flag

# In case you are using any additional library add them here
# If you are using Eigen, assuming it is installed in a standard location, uncomment these lines:
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(KinectFusionLib
  voxel_grid.cpp
  voxel.cpp
  kinect_fusion_utility.cpp
)

add_executable(KinectFusionApp test_tsdf.cpp)

# Linking your KinectFusion library to your executable
target_link_libraries(KinectFusionApp KinectFusionLib)

# This can be run as follows:
# mkdir build
# cd build
# cmake ..
# make
# ./KinectFusionApp