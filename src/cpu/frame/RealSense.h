//
// Created by frankzl on 03.07.19.
// adapted from https://github.com/ParikaGoel/KinectFusion/blob/master/FusionLib/src/KinectSensor.hpp
#pragma once

#include "RealSense.h"
#include <vector>
#include <librealsense2/rs.hpp>
#include "Frame.h"
#include <eigen3/Eigen/Dense>


class RealSensor{
public:

    RealSensor();

    void start();

    void stop();

    std::vector<float> processNextFrame();

    Eigen::Matrix3f getDepthIntrinsics();

    rs2::points getPoints();
    rs2::frameset getFrameset();
    rs2::pointcloud getPointcloud();

private:
    rs2::pipeline m_pipe;
    rs2::pipeline_profile m_profile;

    rs2::frameset m_frameset;

    rs2::pointcloud m_pc;
    rs2::points m_points;
    const rs2::vertex* m_vertices;

    Eigen::Matrix3f K;

    rs2::decimation_filter dec;
    //rs2::disparity_transform depth2disparity;
    rs2::disparity_transform disparity2depth;
    // rs2::spatial_filter spat;
    // rs2::temporal_filter temp;
};