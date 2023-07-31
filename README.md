# KinectFusion-Cool-Edition
https://github.com/amroabuzer/KinectFusion-Cool-Edition
## Introduction
This is our implementation of [KinectFusion](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf) by Richard A. Newcombe et al. 
We aimed to parallelize all parts of the project, and so we make use of cuda for each part. We also aim to provide some CPU implementation at a later date. 

## Dependencies

- **Eigen3** 
- **CUDA 12.2**
- **FreeImage 3**
- (Optional) **Intel® RealSense™ SDK 2.0**

## Instructions
```
mkdir build && cd build
cmake ..
make
cd ..
./Kinect_Fusion.cpp
```
