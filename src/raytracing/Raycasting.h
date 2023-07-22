#ifndef RAYCASTING_H
#define RAYCASTING_H

#include <vector>
#include <Eigen/Dense>
#include "Volume.h"

#define MAX_MARCHING_STEPS 5000
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

	Raycasting(const Volume& _tsdf, const Eigen::Matrix3f& _extrinsics, const Eigen::Vector3f _cameraCenter);

	Vertex castOne(const unsigned int i, const unsigned int j);

	void castAll();

	Eigen::Vector3f computeNormal(const Eigen::Vector3f& p);

	// Eigen::Vector3f getInterpolated(const Ray& r, const double step);

	std::vector<Vertex> getVertices();

private:
	const Eigen::Matrix3f extrinsincs;
	const Eigen::Vector3f cameraCenter;

	Eigen::Matrix3f intrinsics;

	Volume tsdf;

	std::vector<Vertex> vertices;

	unsigned int width;
	unsigned int height;
};


#endif // !RAYCASTING_H