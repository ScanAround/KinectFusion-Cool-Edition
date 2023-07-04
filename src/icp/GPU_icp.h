#pragma once

#include <eigen3/Eigen/Dense>
#include <flann/flann.h>
#include "../frame/Frame_Pyramid.h"


class ICP{
private:

    const Frame_Pyramid F_k;
    const Frame_Pyramid F_k__1;

public:

    ICP(const Frame_Pyramid& F_k, const Frame_Pyramid& F_k__1): F_k(F_k), F_k__1(F_k__1){};
    ~ICP();


    Eigen::Vector4f point_to_plane_solver(const Frame & source, const Frame & target, int iterations, bool cuda, Eigen::Vector4f T_gk_1);

    std::unique_ptr<int> NN_finder(const std::vector<Eigen::Vector3f> & source, const std::vector<Eigen::Vector3f> & target);
};