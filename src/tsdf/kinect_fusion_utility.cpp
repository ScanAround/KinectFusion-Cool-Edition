#include "kinect_fusion_utility.h"

namespace kinect_fusion {

Eigen::Matrix4d utility::getPoseFromTimestamp(const std::string& depthImageFilename, 
                                              const std::string& poseFilePath) {
  // Extract the timestamp from the filename
  std::string filename = depthImageFilename.substr(depthImageFilename.find_last_of("/\\") + 1);
  std::string filenameNoExt = filename.substr(0, filename.find(".png"));

  // Split prefix and suffix from the timestamp
  std::string timestampPrefix = filenameNoExt.substr(0, filenameNoExt.find("."));
  std::string timestampSuffix = filenameNoExt.substr(filenameNoExt.find(".") + 1);

  // Convert prefix and suffix to double
  double timestampPrefixDbl = std::stod(timestampPrefix);
  double timestampSuffixDbl = std::stod(timestampSuffix);

  std::ifstream poseFile(poseFilePath);
  std::string line;

  Eigen::Matrix4d pose;

  double closest_prefix_diff = std::numeric_limits<double>::infinity();
  double closest_suffix_diff = std::numeric_limits<double>::infinity();

  while (std::getline(poseFile, line)) {
    // Skip comments
    if (line.find("#") == 0) {
      continue;
    }

    std::istringstream iss(line);
    double t, tx, ty, tz, qx, qy, qz, qw;

    if (!(iss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
      break; // Error or comment
    }

    // Split prefix and suffix from the timestamp
    std::string poseTimestampStr = std::to_string(t);
    std::string poseTimestampPrefixStr = poseTimestampStr.substr(0, poseTimestampStr.find("."));
    std::string poseTimestampSuffixStr = poseTimestampStr.substr(poseTimestampStr.find(".") + 1);

    // Convert prefix and suffix to double
    double poseTimestampPrefixDbl = std::stod(poseTimestampPrefixStr);
    double poseTimestampSuffixDbl = std::stod(poseTimestampSuffixStr);

    // Compute prefix and suffix differences
    double prefix_diff = std::abs(poseTimestampPrefixDbl - timestampPrefixDbl);
    double suffix_diff = std::abs(poseTimestampSuffixDbl - timestampSuffixDbl);

    // Check if this is the closest timestamp so far
    if (prefix_diff < closest_prefix_diff || 
        (prefix_diff == closest_prefix_diff && suffix_diff < closest_suffix_diff)) {
      closest_prefix_diff = prefix_diff;
      closest_suffix_diff = suffix_diff;

      // This is the pose we want
      Eigen::Vector3d trans(tx, ty, tz);
      Eigen::Quaterniond quat(qw, qx, qy, qz);
      pose.topLeftCorner<3, 3>() = quat.toRotationMatrix();
      pose.topRightCorner<3, 1>() = trans;
      pose(3,0) = pose(3,1) = pose(3,2) = 0.0;
      pose(3,3) = 1.0;
    } else if (prefix_diff > closest_prefix_diff) {
      // The timestamps in the file are in ascending order.
      // If the difference starts to increase, we have passed the closest timestamp.
      break;
    }
  }

  poseFile.close();

  // Return the pose corresponding to the depth map
  return pose;
}

// auto [avgMinDepth, avgMaxDepth, avgAvgDepth] = getDepthRange(directory);
std::tuple<double, double, double> utility::getDepthRange(const std::string& directory) {
    namespace fs = std::filesystem;

    std::vector<float> minDepths, maxDepths, avgDepths;
    for (const auto & entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".png") {
            // Load the depth image
            cv::Mat depthImage = cv::imread(entry.path(), cv::IMREAD_UNCHANGED);

            // Convert to floating point depth in meters
            depthImage.convertTo(depthImage, CV_32F);
            depthImage = depthImage / 5000.0;

            // Mask out zero values (missing data)
            cv::Mat mask = depthImage > 0;
            cv::Mat maskedDepthImage;
            depthImage.copyTo(maskedDepthImage, mask);

            if (cv::countNonZero(mask) > 0) { // Check that there is data
                // Compute minimum and maximum depth
                double minDepth, maxDepth, avgDepth;
                cv::minMaxLoc(maskedDepthImage, &minDepth, &maxDepth, 0, 0, mask);
                avgDepth = cv::mean(maskedDepthImage, mask)[0];

                minDepths.push_back(minDepth);
                maxDepths.push_back(maxDepth);
                avgDepths.push_back(avgDepth);
            }
        }
    }

    // Compute average minimum, maximum and average depth
    double avgMinDepth = std::accumulate(minDepths.begin(), minDepths.end(), 0.0) / 
                         minDepths.size();
    double avgMaxDepth = std::accumulate(maxDepths.begin(), maxDepths.end(), 0.0) / 
                         maxDepths.size();
    double avgAvgDepth = std::accumulate(avgDepths.begin(), avgDepths.end(), 0.0) / 
                         avgDepths.size();

    return std::make_tuple(avgMinDepth, avgMaxDepth, avgAvgDepth);
}

std::vector<std::string> utility::getPngFilesInDirectory(const std::string& directoryPath) {
    DIR* dir;
    struct dirent* ent;
    std::vector<std::string> fileNames;

    if ((dir = opendir(directoryPath.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string fileName = ent->d_name;
            if (fileName.find(".png") != std::string::npos) {  // Only process .png files
                fileNames.push_back(fileName);
            }
        }
        closedir(dir);
    } else {
        throw std::runtime_error("Could not open directory: " + directoryPath);
    }

    // Sort filenames numerically considering the prefix and suffix
    std::sort(fileNames.begin(), fileNames.end(), [](const std::string& a, const std::string& b) {
        size_t dotPosA = a.find(".");
        size_t dotPosB = b.find(".");

        double prefixA = std::stod(a.substr(0, dotPosA));
        double prefixB = std::stod(b.substr(0, dotPosB));

        if (prefixA != prefixB)
            return prefixA < prefixB;

        double suffixA = std::stod(a.substr(dotPosA+1, a.find(".png")-dotPosA-1));
        double suffixB = std::stod(b.substr(dotPosB+1, b.find(".png")-dotPosB-1));

        return suffixA < suffixB;
    });

    return fileNames;
}

Eigen::MatrixXd utility::loadDepthImage(const std::string& filename) {
  // Initialise the FreeImage library
  FreeImage_Initialise();

  FIBITMAP* bitmap = FreeImage_Load(FIF_PNG, filename.c_str(), PNG_DEFAULT);
  if(!bitmap) {
    // De-initialise before throwing an exception
    FreeImage_DeInitialise();
    throw std::runtime_error("Could not load image: " + filename);
  }

  int width = FreeImage_GetWidth(bitmap);
  int height = FreeImage_GetHeight(bitmap);

  Eigen::MatrixXd depth_image(height, width);

  BYTE *bits = (BYTE*)FreeImage_GetBits(bitmap);
  int pitch = FreeImage_GetPitch(bitmap);

  for(int i = 0; i < height; ++i) {
    BYTE *pixel = (BYTE*)bits;
    for(int j = 0; j < width; ++j) {
      // Read 16-bit depth value directly from pixel data
      unsigned short depth_value = *((unsigned short *)pixel);

      // Scale the depth value
      // According to dataset specifications, 5000 units correspond to 1 meter
      depth_image(i, j) = static_cast<double>(depth_value) / 5000.0;

      pixel += 2; // Move to next pixel, each pixel is 2 bytes (16 bits)
    }
    bits += pitch; // Move to next row
  }

  FreeImage_Unload(bitmap);

  // De-initialise the FreeImage library after finishing all operations
  FreeImage_DeInitialise();

  return depth_image;
}


// This function writes the Truncated Signed Distance Field (TSDF) as a text file, with each voxel 
// written as a separate line in the format x y z tsdf_value.
void utility::writeTSDFToFile(const std::string& filePath, kinect_fusion::VoxelGrid& grid) {
  std::ofstream outFile(filePath);
  if (!outFile) {
    std::cerr << "Failed to open file: " << filePath << std::endl;
    return;
  }
  
  for (size_t x = 0; x < grid.getDimX(); ++x) {
    for (size_t y = 0; y < grid.getDimY(); ++y) {
      for (size_t z = 0; z < grid.getDimZ(); ++z) {
        kinect_fusion::Voxel& voxel = grid.getVoxel(x, y, z);
        if (!std::isnan(voxel.tsdfValue) && voxel.tsdfValue!=1){
        outFile << voxel.position(0) << " " << voxel.position(1) << " " << voxel.position(2) 
                << " " << voxel.tsdfValue << "\n";
        }
      }
    }
  }

  outFile.close();
}

}