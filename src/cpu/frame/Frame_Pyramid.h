#ifndef FRAME_PYRAMID_H
#define FRAME_PYRAMID_H
#include "Frame.h"
#include <iostream>
#include <Eigen/Dense>
#include <FreeImage.h> 


class Frame_Pyramid{
public: 
    std::array<Frame * , 3> Depth_Pyramid;

    Frame_Pyramid(FIBITMAP & dib);
    
    Frame_Pyramid(std::string image_dir);

    Frame_Pyramid(std::vector<Eigen::Vector3f> V_gks, std::vector<Eigen::Vector3f> N_gks, Eigen::Matrix4f T_gk, int width = 640, int height = 480);

    ~Frame_Pyramid();

    Eigen::Matrix4f T_gk;

    void set_T_gk(Eigen::Matrix4f T_gk){
        this -> T_gk = T_gk;
        Depth_Pyramid[0] -> T_gk = T_gk;
        Depth_Pyramid[1] -> T_gk = T_gk;
        Depth_Pyramid[2] -> T_gk = T_gk;
    };
};

#endif