# Cuda's Cmake
cmake_minimum_required(VERSION 3.10)
project(KinectFusion)

# Enable CUDA support
enable_language(CUDA)

# Set C++ standard to C++20 for both library and executable
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)  # Set build type to Debug which will automatically add -g flag
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

# In case you are using any additional library add them here
# If you are using Eigen, assuming it is installed in a standard location, uncomment these lines:
find_package(Eigen3 REQUIRED)

# Set CUDA source properties
set_source_files_properties(
    src/gpu/frame/frame.cu
    src/gpu/raytracing/raycasting.cu
    src/gpu/icp/icp.cu
    src/gpu/tsdf/tsdf.cu
    PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)

# Create the library target for KinectFusionLib
add_library(KinectFusionLib
    src/gpu/frame/frame.cu
    src/gpu/raytracing/raycasting.cu
    src/gpu/icp/icp.cu
    src/gpu/tsdf/tsdf.cu
    src/cpu/tsdf/voxel_grid.cpp
    src/cpu/tsdf/voxel.cpp
    src/cpu/frame/Frame_Pyramid.cpp
    src/cpu/tsdf/kinect_fusion_utility.cpp
    src/Kinect_Fusion.cpp
)

# Set the C++ standard for the library target
target_compile_features(KinectFusionLib PUBLIC cxx_std_20)

# Link Eigen library to the library target
target_include_directories(KinectFusionLib PRIVATE ${EIGEN3_INCLUDE_DIR})

# Find FreeImage library
find_library(FREEIMAGE_LIBRARIES freeimage)

# Create the executable target for KinectFusionApp_Cuda
add_executable(KinectFusionApp_Cuda
    src/Kinect_Fusion.cpp
)

# Set the C++ standard for the executable target
target_compile_features(KinectFusionApp_Cuda PUBLIC cxx_std_20)

# Link your KinectFusion library and FreeImage library to the executable target
target_link_libraries(KinectFusionApp_Cuda KinectFusionLib ${FREEIMAGE_LIBRARIES})

# Suppress deprecated warnings and some others
# Narrowing warnings should be solved ideally
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(KinectFusionLib PRIVATE -Wno-deprecated -Wno-ignored-attributes -Wno-narrowing)
    target_compile_options(KinectFusionApp_Cuda PRIVATE -Wno-deprecated -Wno-ignored-attributes -Wno-narrowing)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(KinectFusionLib PRIVATE -Wno-deprecated-enum-enum-conversion -Wno-ignored-attributes -Wno-c++11-narrowing)
    target_compile_options(KinectFusionApp_Cuda PRIVATE -Wno-deprecated-enum-enum-conversion -Wno-ignored-attributes -Wno-c++11-narrowing)
endif()

# This can be run as follows:
# mkdir build
# cd build
# cmake ..
# make
# ./KinectFusionApp_Cuda
