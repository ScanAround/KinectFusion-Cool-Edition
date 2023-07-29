cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(kinect LANGUAGES C CXX CUDA)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../Exercises/Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
# Use pre-compiled libraries on Windows
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    endif()
endif(WIN32)
# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

# Define header and source files
set(HEADERS
    src/cpu/frame/Frame.h
src/cpu/frame/Frame_Pyramid.h
    src/cpu/icp/ICP.h
src/cpu/mesher/Marching_Cubes.h
src/cpu/mesher/Lookup_Tables.h
src/cpu/raytracing/Raycasting.h
)

set(SOURCES
# src/gpu/frame/frame.cu
src/gpu/raytracing/raycasting.cu
# src/gpu/icp/icp.cu
src/gpu/tsdf/tsdf.cu
src/cpu/tsdf/voxel_grid.cpp
# src/cpu/tsdf/voxel_grid.cpp
# src/cpu/frame/Frame.cpp
src/cpu/tsdf/voxel.cpp
src/cpu/frame/Frame_Pyramid.cpp
src/cpu/tsdf/kinect_fusion_utility.cpp
src/cpu/tsdf/test_tsdf.cpp
)

link_directories(${FreeImage_LIBRARY_DIR})
add_executable(kinect ${HEADERS} ${SOURCES})
target_link_libraries(kinect freeimage Eigen3::Eigen )
target_include_directories(kinect PUBLIC  ${FreeImage_INCLUDE_DIR} )

# Visual Studio properties
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT kinect)
set_property(TARGET kinect PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)
set_property(TARGET kinect PROPERTY CUDA_ARCHITECTURES native)
set_target_properties(kinect 
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
    # On Windows copy dll to executable folder
    add_custom_command(TARGET kinect POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FreeImage_LIBRARY_DIR}/FreeImage.dll"
            ${CMAKE_BINARY_DIR})
    	
    