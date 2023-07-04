#pragma once

#include <eigen3/Eigen/Dense>
#include <flann/flann.h>
#include "../frame/Frame_Pyramid.h"


class ICP{
private:

    const Frame_Pyramid F_k;
    const Frame_Pyramid F_k__1;
    const double convergence_threshold;

public:

    ICP(const Frame_Pyramid& F_k, const Frame_Pyramid& F_k__1, const double convergence_threshold): F_k(F_k), F_k__1(F_k__1), convergence_threshold(convergence_threshold){};
    ~ICP();


    Eigen::Vector4f point_to_plane_solver(Frame & source, Frame & target, int iterations, bool cuda);

    int * NN_finder(Eigen::Vector4f source_transformation, const Frame & source, const Frame & target);
};