#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>
#include <FreeImage.h>
#include <dirent.h>
#include <algorithm>

namespace kinect_fusion {

class utility {
  private:

  public:

  // This function takes the filename of a depth map and the path to the pose file as input and 
  // returns the corresponding pose as an Eigen::Matrix4d. It can be called as
  // "Eigen::Matrix4d pose = getPoseFromTimestamp("1305031102.160407.png", "groundtruth.txt");"
  static Eigen::Matrix4d getPoseFromTimestamp(const std::string& depthImageFilename, 
                                              const std::string& poseFilePath);

  // This function loops over all depth images in a directory and compute the average minimum and 
  // maximum depths and the average depth.
  static std::tuple<double, double, double> getDepthRange(const std::string& directory);

  // This function numerically / chronologically sorts the depth maps in a given directory. It 
  // first compares the prefixes (before the '.') of each filename. If the prefixes are the same, 
  // then we compare the suffixes (after the '.').
  static std::vector<std::string> getPngFilesInDirectory(const std::string& directoryPath);

  // This function loads depth image and returns it as a Eigen::MatrixXd.
  static Eigen::MatrixXd loadDepthImage(const std::string& filename);

  static void writeTSDFToFile(const std::string& filePath, const kinect_fusion::VoxelGrid& grid) {

};

}