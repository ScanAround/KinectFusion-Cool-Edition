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

int main(){
    //sanity check
    FreeImage_Initialise();
    const char * depth_map_dir = "/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png";
    
    Frame_Pyramid * frame1 = new Frame_Pyramid(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir), depth_map_dir));
    FreeImage_Save(FREE_IMAGE_FORMAT::FIF_PNG, frame1-> Depth_Pyramid[0] ->filtered_dib ,"/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/dummy_shiz/level0.png");
    FreeImage_Save(FREE_IMAGE_FORMAT::FIF_PNG, frame1-> Depth_Pyramid[1] ->filtered_dib ,"/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/dummy_shiz/level1.png");
    FreeImage_Save(FREE_IMAGE_FORMAT::FIF_PNG, frame1-> Depth_Pyramid[2] ->filtered_dib ,"/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/dummy_shiz/level2.png");
    
    frame1 -> Depth_Pyramid[2] -> save_off_format("/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/vertices_level2.obj");

}