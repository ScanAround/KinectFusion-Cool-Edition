#include "Frame_Pyramid.h"
#include "Frame.h"
#include "Frame.cpp"
#include <iostream>
#include <FreeImage.h>
#include <eigen3/Eigen/Dense>
#include <cmath>

Frame_Pyramid::Frame_Pyramid(FIBITMAP & dib){

    Depth_Pyramid[0] = new Frame(dib);
    

}

void Frame_Pyramid::SubSampler()
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
