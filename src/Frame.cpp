#include "Frame.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <FreeImage.h>

Frame::Frame(FIBITMAP & dib): dib(FreeImage_ConvertToFloat(&dib)){
    
    width = FreeImage_GetWidth(this->dib);
    height = FreeImage_GetHeight(this->dib);
    
    Depth_k = new float[width*height];
    std::memcpy(Depth_k, FreeImage_GetBits(this->dib), width*height);
    K_calibration  <<  525.0f, 0.0f, 319.5f,
                        0.0f, 525.0f, 239.5f,
                        0.0f, 0.0f, 1.0f;
}

Frame::~Frame(){
    if(dib != nullptr){delete dib;}
    if(Depth_k != nullptr){delete Depth_k;}
}

Frame::Frame(const Frame & from_other): Depth_k(from_other.Depth_k){
    if(!from_other.V_k.empty() and !from_other.M_k.empty() and !from_other.N_k.empty()){
        V_k = from_other.V_k;
        N_k = from_other.N_k;
        M_k = from_other.M_k;
    }
    else{
        throw std::logic_error("Either V_k vector, N_k vector, or M_k vectors are not initialized");
    }
}

std::vector<Eigen::Vector3f> Frame::calculate_Vks(){
    
    Eigen::Matrix3f K_calibration_inverse = K_calibration.inverse();
    
    Eigen::Vector3f u_dot;

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            u_dot << j, i ,1;
            //dividing by 5000 since scaled by that factor and multiply by 2^16 since stored as 16 bit monochrome images https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
            Eigen::Vector3f ans = (Depth_k[i*width + j] * 1.0f/5000.0f *256 * 256 )* K_calibration_inverse *  u_dot; 
            V_k.push_back(ans);
            // std::cout << ans[0] << ", " << ans[1] << ", " << ans[2] <<std::endl;
        }
    }

    return V_k;
}

std::vector<Eigen::Vector3f> Frame::calculate_Nks(){
    if(!V_k.empty()){
        for(int i = 0; i < height-1; i++){
            for(int j = 0; j < width-1; j++){
               Eigen::Vector3f ans =(V_k[i*width + j+1] - V_k[(i)*width + j]).cross((V_k[(i+1)*width + j] - V_k[(i)*width + j]));
               ans.normalize();
               N_k.push_back(ans);
               std::cout << ans[0] << ", " << ans[1] << ", " << ans[2] <<std::endl;
            }
        }
    }
    else{
        throw std::logic_error("Your V_ks haven't been calculated yet :/");
    }
    return V_k;
}

void Frame::process_image(){
    calculate_Vks();
    calculate_Nks();
}

int main(){
    //sanity check
    FreeImage_Initialise();
    const char * depth_map_dir = "/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png";
    
    Frame * frame1 = new Frame(*FreeImage_Load(FreeImage_GetFileType(depth_map_dir), depth_map_dir));
    
    frame1->process_image();

}