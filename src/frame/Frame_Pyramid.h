#ifndef FRAME_PYRAMID_H
#define FRAME_PYRAMID_H
#include "Frame.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <FreeImage.h> 


class Frame_Pyramid{
public: 
    std::array<Frame * , 3> Depth_Pyramid;

    Frame_Pyramid(FIBITMAP & dib);

    ~Frame_Pyramid();

    Frame_Pyramid(const Frame_Pyramid & from_other) {};

    Frame_Pyramid & operator=(const Frame_Pyramid & from_other);

    Frame_Pyramid(Frame_Pyramid && from_other) {};
    
    Frame_Pyramid &operator=(Frame_Pyramid && from_other);

    Eigen::Matrix4f T_gk;

    void set_T_gk(Eigen::Matrix4f& T_gk){
        Depth_Pyramid[0] -> T_gk = T_gk;
        Depth_Pyramid[1] -> T_gk = T_gk;
        Depth_Pyramid[2] -> T_gk = T_gk;
    };
};

#endif