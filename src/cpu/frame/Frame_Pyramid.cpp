#include "Frame_Pyramid.h"
#include "Frame.h"
#include <iostream>
#include <FreeImage.h>
#include <eigen3/Eigen/Dense>
#include <cmath>

Frame_Pyramid::Frame_Pyramid(FIBITMAP & dib){

    T_gk << Eigen::Matrix4f::Identity();

    Depth_Pyramid[0] = new Frame(dib, T_gk);
    Depth_Pyramid[0] -> process_image();

    Depth_Pyramid[1] = new Frame(*FreeImage_Rescale(Depth_Pyramid[0] -> filtered_dib, 
                                                    Depth_Pyramid[0] -> width/2,
                                                    Depth_Pyramid[0] -> height/2,
                                                    FILTER_BOX), T_gk, 2.0f);
    Depth_Pyramid[1] -> process_image(0.01, 2.0, 10);

    Depth_Pyramid[2] = new Frame(*FreeImage_Rescale(Depth_Pyramid[1] -> filtered_dib, 
                                                    Depth_Pyramid[1] -> width/2,
                                                    Depth_Pyramid[1] -> height/2,
                                                    FILTER_BOX), T_gk, 4.0f);
    Depth_Pyramid[2] -> process_image(0.01, 1.0, 5);
}

Frame_Pyramid::Frame_Pyramid(std::string image_dir_s){

    const char * image_dir = image_dir_s.c_str();

    T_gk << Eigen::Matrix4f::Identity();

    Depth_Pyramid[0] = new Frame(image_dir, T_gk);
    Depth_Pyramid[0] -> process_image();

    Depth_Pyramid[1] = new Frame(*FreeImage_Rescale(Depth_Pyramid[0] -> filtered_dib, 
                                                    Depth_Pyramid[0] -> width/2,
                                                    Depth_Pyramid[0] -> height/2,
                                                    FILTER_BOX), T_gk, 2.0f);
    Depth_Pyramid[1] -> process_image(0.01, 2.0, 10);

    Depth_Pyramid[2] = new Frame(*FreeImage_Rescale(Depth_Pyramid[1] -> filtered_dib, 
                                                    Depth_Pyramid[1] -> width/2,
                                                    Depth_Pyramid[1] -> height/2,
                                                    FILTER_BOX), T_gk, 4.0f);
    Depth_Pyramid[2] -> process_image(0.01, 1.0, 5);
}

Frame_Pyramid::Frame_Pyramid(Frame & raytracing_frame){

    T_gk << raytracing_frame.T_gk;

    Depth_Pyramid[0] = &raytracing_frame;
    Depth_Pyramid[1] = &raytracing_frame;
    Depth_Pyramid[2] = &raytracing_frame;
    
}

Frame_Pyramid::~Frame_Pyramid(){
    delete Depth_Pyramid[0];
    delete Depth_Pyramid[1];
    delete Depth_Pyramid[2];
}