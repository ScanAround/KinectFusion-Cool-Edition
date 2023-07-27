cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(KinectFusion)

# Options
set(LIBRARY_DIR C:/Users/marcw/Desktop/BMC_TUM/sose23/3dsmc/Exercises/Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(OpenCV_DIR C:/lib/install/opencv CACHE PATH "Path to installed OpenCV")

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

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

set(HEADERS 
../mesher/Marching_Cubes.h
../mesher/Lookup_Tables.h
../raytracing/Raycasting.h
../raytracing/Ray.h
)

set(SOURCES
test_tsdf.cpp
voxel_grid.cpp
voxel.cpp
kinect_fusion_utility.cpp
../frame/Frame.cpp
../raytracing/Raycasting.cpp
../raytracing/Ray.cpp
)

link_directories(${FreeImage_LIBRARY_DIR})
add_executable(raycasting ${HEADERS} ${SOURCES})
target_include_directories(raycasting PUBLIC ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR})
target_link_libraries(raycasting general Eigen3::Eigen freeimage ${OpenCV_LIBS})

if(WIN32)
    # Visual Studio properties
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT raycasting)
    set_property(TARGET raycasting PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

    # On Windows copy dll to executable folder
    add_custom_command(TARGET raycasting POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FreeImage_LIBRARY_DIR}/FreeImage.dll"
            ${CMAKE_BINARY_DIR})
endif(WIN32)