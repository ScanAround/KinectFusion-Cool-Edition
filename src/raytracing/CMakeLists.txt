cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(raycasting)

set(LIBRARY_DIR C:/Users/marcw/Desktop/BMC_TUM/sose23/3dsmc/Exercises/Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)

set(HEADERS
	ImplicitSurface.h
	Ray.h
	Raycasting.h
)

set(SOURCES
	main.cpp
)

add_executable(raycasting ${HEADERS} ${SOURCES})
target_include_directories(raycasting PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(raycasting Eigen3::Eigen)

# Visual Studio properties
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT raycasting)
set_property(TARGET raycasting PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)