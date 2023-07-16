#include "Frame_Pyramid.h"
#include "Frame.h"
#include <iostream>
#include <FreeImage.h>
#include <eigen3/Eigen/Dense>
#include <cmath>

Frame_Pyramid::Frame_Pyramid(FIBITMAP & dib){

    Depth_Pyramid[0] = new Frame(dib);
    Depth_Pyramid[0] -> process_image();

    Depth_Pyramid[1] = new Frame(*FreeImage_Rescale(Depth_Pyramid[0] -> filtered_dib, 
                                                    Depth_Pyramid[0] -> width/2,
                                                    Depth_Pyramid[0] -> height/2,
                                                    FILTER_BOX), 2.0f);
    Depth_Pyramid[1] -> process_image(0.01, 2.0, 10);

    Depth_Pyramid[2] = new Frame(*FreeImage_Rescale(Depth_Pyramid[1] -> filtered_dib, 
                                                    Depth_Pyramid[1] -> width/2,
                                                    Depth_Pyramid[1] -> height/2,
                                                    FILTER_BOX), 4.0f);
    Depth_Pyramid[2] -> process_image(0.01, 1.0, 5);
}

Frame_Pyramid::~Frame_Pyramid(){
    delete Depth_Pyramid[0];
    delete Depth_Pyramid[1];
    delete Depth_Pyramid[2];
}