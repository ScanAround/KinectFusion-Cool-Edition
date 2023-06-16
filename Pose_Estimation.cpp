#include "Pose_Estimation.h"
#include <iostream>
#include <FreeImage.h>
#include <eigen3/Eigen/Dense>


FIBITMAP Pose_Estimator::SubSampler(FIBITMAP * D_k)
{
    int w = FreeImage_GetWidth(D_k);
    int h = FreeImage_GetHeight(D_k);

    BYTE * bits = FreeImage_GetBits(D_k);
    BYTE * subsampled_bits = new BYTE[w*h/4];

    for (int x = 0; x<w-1 ; x+2){
        for(int y = 0; y<h-1; y+2){
            subsampled_bits[x*y/2 + y/2] = (bits[x*y+y] + bits[x*y+y+1] + bits[x*y+y+2] + bits[x*y+y+3])/4;
        }
    }
    
}

int main(){


    //sanity check

    FreeImage_Initialise();
    const char * depth_map_dir = "/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png";
    FIBITMAP * depth_map = FreeImage_Load(FreeImage_GetFileType(depth_map_dir), depth_map_dir);
    BYTE* bits = FreeImage_GetBits(depth_map);

    std::vector<Eigen::Vector3f> V_k;
    std::vector<Eigen::Vector3f> N_k;
    
    V_k.push_back(Eigen::Vector3f(1.0f, 2.0f, 3.0f));
    N_k.push_back(Eigen::Vector3f(4.0f, 5.0f, 6.0f));
}