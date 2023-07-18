#pragma once

#include <eigen3/Eigen/Dense>
#include "../frame/Frame_Pyramid.h"
#include <FreeImage.h>

class ICP{

public:

    ICP(const Frame_Pyramid& F_k, const Frame_Pyramid& F_k__1, const float distance_threshold, const float angle_threshold): 
    F_k(F_k), 
    F_k__1(F_k__1), 
    distance_threshold(distance_threshold),
    angle_threshold(angle_threshold){};
    
    ~ICP();

    Eigen::Matrix4f point_to_plane_solver(Frame & curr_frame, Frame & prev_frame, int iterations, bool cuda);

    Eigen::Matrix4f pyramid_ICP(Frame_Pyramid & curr_frame, Frame_Pyramid & prev_frame, bool cuda);

    void correspondence_finder(Eigen::Matrix4f T_curr_frame, Frame & curr_frame, Frame & prev_frame, std::vector<std::pair<int, int>>& matches);

private:

    const Frame_Pyramid F_k;
    const Frame_Pyramid F_k__1;
    const float distance_threshold;
    const float angle_threshold;
};