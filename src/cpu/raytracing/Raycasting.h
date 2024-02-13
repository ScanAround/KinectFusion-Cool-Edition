#ifndef RAYCASTING_H
#define RAYCASTING_H

#include <eigen3/Eigen/Dense>
#include <vector>
// #include "Volume.h"
// #include "Ray.h"
#include "../../cpu/tsdf/voxel_grid.h"

#define MAX_MARCHING_STEPS 20000
#define MINF -std::numeric_limits<float>::infinity()

struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	Eigen::Vector3f position;
	Eigen::Vector3f normal;
};

class Raycasting
{
public:

	// Raycasting(const Volume& _tsdf, const Eigen::Matrix3f& _extrinsics, const Eigen::Vector3f _cameraCenter);

	Raycasting(kinect_fusion::VoxelGrid& _tsdf, const Eigen::Matrix3f& _extrinsics, const Eigen::Vector3f _cameraCenter);

	~Raycasting();

	Vertex castOne(const unsigned int i, const unsigned int j);

	void castAll();

	void castAllCuda();

	Eigen::Vector3f computeNormal(const Eigen::Vector3f& p);

	// Eigen::Vector3f getInterpolated(const Ray& r, const double step);

	std::vector<Eigen::Vector3f> getVertices();
	std::vector<Eigen::Vector3f> getNormals();

	void writePointCloud(const std::string& filename);

	void freeCuda();

private:
	const Eigen::Matrix3f extrinsincs;
	const Eigen::Vector3f cameraCenter;

	Vertex *vertices;
	// std::vector<Vertex> vertices;

	Eigen::Matrix3f intrinsics;

	// Volume tsdf;
	kinect_fusion::VoxelGrid* tsdf;

	unsigned int width;
	unsigned int height;
};


#endif // !RAYCASTING_H