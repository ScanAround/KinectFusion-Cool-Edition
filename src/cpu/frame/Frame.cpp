#include "Frame.h"
#include <iostream>
#include <FreeImage.h>
#include <fstream>
#include <math.h>
#include <array>
#include <vector>
#define MAXTHRESHOLD 10
#define MINF -std::numeric_limits<float>::infinity()


inline double Frame::N_sigma(const float& sigma, const float &t){
    return exp(-std::pow(t,2)*std::pow(sigma,-2));
}
FIBITMAP * Frame::Apply_Bilateral(const float & sigma_r, const float & sigma_s, const int & filter_size){
    //need to clean this shit up 
    Eigen::Matrix3f K_calibration_inverse = K_calibration.inverse();
    FIBITMAP * result;
    result = FreeImage_Allocate(width, height, 8); // monochrome image therefore 8 bytes
    BYTE * image_data = FreeImage_GetBits(result);
    for(int i = static_cast<int>(filter_size/2) ; i < height-(static_cast<int>(filter_size/2)); ++i){
        for(int j = static_cast<int>(filter_size/2); j < width-(static_cast<int>(filter_size/2)); ++j){
            float sum = 0.0f;
            float normalizing_constant = 0.0f;
            for(int q = 0; q < filter_size*filter_size; ++q){
                if ( Raw_k[(i + (q % filter_size - static_cast<int>(filter_size/2))) * width + j + (int)q / filter_size -static_cast<int>(filter_size/2)] <= 0.0f ||  Raw_k[(i + (q % filter_size - static_cast<int>(filter_size/2))) * width + j + (int)q / filter_size -static_cast<int>(filter_size/2)] == MINF) {
					continue;
                    }
                else{
                sum += N_sigma(sigma_s,  sqrt(std::pow( -(int) q/filter_size + static_cast<int>(filter_size/2), 2) + std::pow( -(q % filter_size) + static_cast<int>(filter_size/2), 2)))
                * N_sigma(sigma_r, Raw_k[i*width + j]-Raw_k[(i + (q % filter_size - static_cast<int>(filter_size/2))) * width + j + (int)q / filter_size -static_cast<int>(filter_size/2)])
                * Raw_k[(i + (q % filter_size - static_cast<int>(filter_size/2))) * width + j + (int)q / filter_size -static_cast<int>(filter_size/2)];
      
                normalizing_constant += N_sigma(sigma_s,  sqrt(std::pow( -(int) q/filter_size + static_cast<int>(filter_size/2), 2) + std::pow( -(q % filter_size) + static_cast<int>(filter_size/2), 2)))
                * N_sigma(sigma_r, Raw_k[i*width + j]-Raw_k[(i + (q % filter_size - static_cast<int>(filter_size/2))) * width + j + (int)q / filter_size -static_cast<int>(filter_size/2)]);
                }
            }
            
            Depth_k[i*width + j] = sum/normalizing_constant * 255.0f * 255.0f;
            image_data[i*width + j] = static_cast<BYTE>(Depth_k[i*width + j] / 255.0f);  // done to see filtered image
        }
    }
    return result;
}

void Frame::save_off_format(const std::string & where_to_save){
    std::ofstream OffFile(where_to_save);
    for(auto i : M_k1){
        if(abs(V_k[i][0]) < MAXTHRESHOLD){
            OffFile << "v " << V_k[i][0] << " " << V_k[i][1] << " " << V_k[i][2] << std::endl; 
            if(!std::isnan(N_k[i][0]) && !std::isnan(N_k[i][1]) && !std::isnan(N_k[i][2])){
                OffFile << "vn " << N_k[i][0] << " " << N_k[i][1] << " " << N_k[i][2] << std::endl;
            }
            else{
                OffFile << "vn " << 0 << " " << 0 << " " << 0 << std::endl;
            } 
        }
    }
    OffFile.close();
}

void Frame::save_G_off_format(const std::string & where_to_save)
{
        std::ofstream OffFile(where_to_save);
        // this -> apply_G_transform();
        for(unsigned int i = 0; i < width * height; ++i){
            if(abs(V_gk[i][0]) < MAXTHRESHOLD){
                if (V_gk[i][0] != MINF)
                {
                    OffFile << "v " << V_gk[i][0] << " " << V_gk[i][1] << " " << V_gk[i][2] << std::endl; 
                    if(!std::isnan(N_gk[i][0]) && !std::isnan(N_gk[i][1]) && !std::isnan(N_gk[i][2])){
                        OffFile << "vn " << N_gk[i][0] << " " << N_gk[i][1] << " " << N_gk[i][2] << std::endl;
                    }
                    else{
                        OffFile << "vn " << 0 << " " << 0 << " " << 0 << std::endl;
                    } 
                }
            }
        }
        OffFile.close();
    }

Frame::Frame(FIBITMAP & dib, Eigen::Matrix4f T_gk, float sub_sampling_rate): 
dib(FreeImage_ConvertToFloat(&dib)), T_gk(T_gk){
    
    width = FreeImage_GetWidth(this->dib);
    height = FreeImage_GetHeight(this->dib);
    
    Depth_k = new float[width*height]; // have to rescale according to the data 
    Raw_k = (float *) FreeImage_GetBits(this->dib) ; // have to rescale according to the data 
    
    K_calibration  <<  525.0f / sub_sampling_rate, 0.0f, 319.5f / sub_sampling_rate,
                        0.0f, 525.0f / sub_sampling_rate, 239.5f/ sub_sampling_rate,
                        0.0f, 0.0f, 1.0f;
}

Frame::Frame(const char * image_dir, Eigen::Matrix4f T_gk, float sub_sampling_rate): 
dib(FreeImage_ConvertToFloat(FreeImage_Load(FreeImage_GetFileType(image_dir), image_dir))){
    FreeImage_Initialise();
    width = FreeImage_GetWidth(this->dib);
    height = FreeImage_GetHeight(this->dib);
    
    Depth_k = new float[width*height]; // have to rescale according to the data 
    Raw_k = (float *) FreeImage_GetBits(this->dib) ; // have to rescale according to the data 
    K_calibration  <<  525.0f / sub_sampling_rate, 0.0f, 319.5f / sub_sampling_rate,
                        0.0f, 525.0f / sub_sampling_rate, 239.5f/ sub_sampling_rate,
                        0.0f, 0.0f, 1.0f;
    this -> T_gk = T_gk;
    FreeImage_DeInitialise();
}

Frame::Frame(std::vector<Eigen::Vector3f> V_gks, std::vector<Eigen::Vector3f> N_gks, Eigen::Matrix4f T_gk, int width, int height):
width(width), height(height), T_gk(T_gk), V_gk(V_gks), N_gk(N_gks){
    K_calibration  <<  525.0f , 0.0f, 319.5f,
                        0.0f, 525.0f, 239.5f,
                        0.0f, 0.0f, 1.0f;
    
}

Frame::~Frame(){
    if(dib != nullptr){delete dib;}
    if(Depth_k != nullptr){delete[] Depth_k;}
    // if(Raw_k != nullptr){delete Raw_k;}
}

std::vector<Eigen::Vector3f> Frame::calculate_Vks(){
    
    Eigen::Matrix3f K_calibration_inverse = K_calibration.inverse();
    
    Eigen::Vector3f u_dot;
    int counter = 0;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            u_dot << j, i ,1;
            if(Depth_k[i*width + j] == MINF || Depth_k[i*width + j] <= 0.0f){
                V_k.push_back(Eigen::Vector3f(MINF, MINF, MINF));
                M_k0.push_back(i*width+j);
            }
            else{
                
                //dividing by 5000 since scaled by that factor https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
                Eigen::Vector3f ans = (Depth_k[i*width + j]/ 5000.0f)* K_calibration_inverse *  u_dot; 
                V_k.push_back(ans);
                M_k1.push_back(i*width+j);
                
            }
            // std::cout << ans[0] << ", " << ans[1] << ", " << ans[2] <<std::endl;
        }
    }
    return V_k;
}

std::vector<Eigen::Vector3f> Frame::calculate_Nks(){
    if(!V_k.empty()){
        for(int i = 0; i < height - 1; i++){
            for(int j = 0; j < width - 1; j++){
                Eigen::Vector3f ans =(V_k[i*width + j+1] - V_k[(i)*width + j]).cross((V_k[(i+1)*width + j] - V_k[(i)*width + j]));
                ans.normalize();
                N_k.push_back(ans);
            }
            Eigen::Vector3f ans =(V_k[i*width + (width-1) - 1] - V_k[(i)*width + (width-1)]).cross((V_k[(i+1)*width + (width-1)] - V_k[(i)*width + (width-1)]));
            ans.normalize();
            N_k.push_back(ans);
        }
    }
    else{
        throw std::logic_error("Your V_ks haven't been calculated yet :/");
    }
    return N_k;
}

void Frame::process_image(float sigma_r , float sigma_s ,  int filter_size, bool apply_bilateral){
    if(apply_bilateral){
        filtered_dib = Apply_Bilateral(sigma_r, sigma_s, filter_size);
    }
    else{
        filtered_dib = dib;
        Depth_k = Raw_k;
    }
    // FreeImage_Save(FREE_IMAGE_FORMAT::FIF_PNG, filtered_image,"/mnt/c/Users/asnra/Desktop/Coding/KinectFusion/KinectFusion-Cool-Edition/data/dummy_shiz/bilateral_filter.png");
    calculate_Vks();
    calculate_Nks();
}