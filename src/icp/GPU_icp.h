#pragma once

#include <eigen3/Eigen/Dense>
#include "../frame/Frame_Pyramid.h"
#include "NN.h"
#include <FreeImage.h>

class ICP{

public:

    ICP(const Frame_Pyramid& F_k, const Frame_Pyramid& F_k__1, const double convergence_threshold): 
    F_k(F_k), 
    F_k__1(F_k__1), 
    convergence_threshold(convergence_threshold){};
    
    ~ICP();

    Eigen::Matrix4f point_to_plane_solver(Frame & source, Frame & target, int iterations, bool cuda);

    void NN_finder(Eigen::Matrix4f source_transformation, Frame & source, const Frame & target, std::vector<Match>& matches);

private:

    const Frame_Pyramid F_k;
    const Frame_Pyramid F_k__1;
    const double convergence_threshold;
    NN_flann* my_NN = new NN_flann(1, 0.005f); // should be tweaked later to improve performance
};