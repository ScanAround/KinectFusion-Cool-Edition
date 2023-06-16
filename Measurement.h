#ifndef MEASUREMENT_H
#define MEASUREMENT_H
#include "Pose_Estimation.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <FreeImage.h>

class Measurer{

private:

// Rotation Matrix from frame k
Eigen::Matrix3f R_k;

// Calibration Matrix K 
Eigen::Matrix3f K;

std::vector<Eigen::Vector3f> Find_Vs();
std::vector<Eigen::Vector3f> Find_Ns();

public:

Measurer(Eigen::Matrix3f R_k, );

std::vector<Eigen::Vector3f> get_Vs();
std::vector<Eigen::Vector3f> get_Ns();

};

#endif