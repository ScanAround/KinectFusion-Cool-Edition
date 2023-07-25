#include "../../cpu/frame/Frame.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <FreeImage.h>
#include <fstream>
#include <math.h>
#include <array>
#include <vector>

#define MAXTHRESHOLD 10

__global__ 
void apply_bilateral(){}

__global__ 
void calculate_Vks(Eigen::Matrix3f K_i,
                   Eigen::Vector3f *dV_k,
                   float *Depth_k, int *dMk_0, int *dMk_1,
                   int width, int height){
  
  int id_x = blockIdx.x; // each block is a column
  int id_y = threadIdx.y; // each thread is a pixel
  Eigen::Vector3f u_dot;

  if(id_x < width && id_y < height){
    u_dot << id_x, id_y, 1;
    if(Depth_k[id_y * width + id_x] == -INFINITY || Depth_k[id_y * width + id_x] <= 0.0f){
        dV_k[id_y * width + id_x] = Eigen::Vector3f(-INFINITY, -INFINITY, -INFINITY);
        dMk_0[id_y * width + id_x] = id_y * width + id_x;
    }  
    else{
         dV_k[id_y * width + id_x] = Depth_k[id_y * width + id_x]/5000.0f * K_i * u_dot;
         dMk_1[id_y * width + id_x] = id_y * width + id_x;
    }
  }
}

__global__ 
void calculate_Nks(Eigen::Vector3f *dV_k,
                   Eigen::Vector3f *dN_k,
                   int width, int height){}

