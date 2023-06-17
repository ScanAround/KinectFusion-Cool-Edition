#include "Pose_Estimation.h"
#include <iostream>
#include <FreeImage.h>
#include <eigen3/Eigen/Dense>
#include <cmath>

// FIBITMAP * Pose_Estimator::get_D_k() const{
//     return this->D_k;
// }

FIBITMAP * Pose_Estimator::SubSampler(FIBITMAP * D_k)
{
    int w = FreeImage_GetWidth(D_k);
    int h = FreeImage_GetHeight(D_k);

    BYTE * bits = FreeImage_GetBits(D_k);
    
    
    // Find the block average of the image and subsample

    // First we do a baseline

    FIBITMAP * downsampled_image = FreeImage_Rescale(D_k, w/2, h/2, FILTER_BOX);

        // Find the block average of the image and only include pixels within 3_sigma_r of the center value (attempt)

        // BYTE * subsampled_bits = new BYTE[w*h/4];

        // for (int x = 1; x<w-1 ; x++){
        //     for(int y = 1; y<h-1 ; y++){
        //         float average = (bits[(x-1)*(y-1)+(y-1)] + bits[x*y+y+1] + bits[x*y+y+2] + bits[x*y+y+3]+ bits[x*y+y+3]+ bits[x*y+y+3])/9;
        //         float 3_sigma_r = 3 * std::sqrt() 
        //         subsampled_bits[x*y/2 + y/2] = (bits[x*y+y] + bits[x*y+y+1] + bits[x*y+y+2] + bits[x*y+y+3])/9;
        //     }
        // }

    return downsampled_image;
}

FIBITMAP * Pose_Estimator::Pyramid_Generator()
{

    const FIBITMAP * D_k = this->D_k;

    FIBITMAP * Pyramid = new FIBITMAP[3];
    Pyramid[0] = *D_k;
    Pyramid[1] = *this->SubSampler(&Pyramid[0]);
    Pyramid[2] = *this->SubSampler(&Pyramid[1]);

    return Pyramid;
}

int main(){


    //sanity check

    FreeImage_Initialise();
    const char * depth_map_dir = "/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png";
    FIBITMAP * depth_map = FreeImage_Load(FreeImage_GetFileType(depth_map_dir), depth_map_dir);
    // BYTE* bits = FreeImage_GetBits(depth_map);
    
    std::vector<Eigen::Vector3f> * V_k = new std::vector<Eigen::Vector3f>;
    std::vector<Eigen::Vector3f> * N_k = new std::vector<Eigen::Vector3f>;
    
    V_k->push_back(Eigen::Vector3f(1.0f, 2.0f, 3.0f));
    N_k->push_back(Eigen::Vector3f(4.0f, 5.0f, 6.0f));

    Pose_Estimator * Pose1 = new Pose_Estimator(V_k , N_k , depth_map );

    FIBITMAP * pyramid = Pose1->Pyramid_Generator();

    FreeImage_Save(FreeImage_GetFileType(depth_map_dir), &pyramid[0],"/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/dummy_shiz/level0.png");
    FreeImage_Save(FreeImage_GetFileType(depth_map_dir), &pyramid[1],"/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/dummy_shiz/level1.png");
    FreeImage_Save(FreeImage_GetFileType(depth_map_dir), &pyramid[2],"/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/dummy_shiz/level2.png");

}