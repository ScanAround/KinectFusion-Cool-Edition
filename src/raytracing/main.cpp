#include <iostream>

#include <Eigen/Dense>
#include "ImplicitSurface.h"
#include "Volume.h"

// TODO: choose optimal truncation value
#define TRUNCATION 1.0
#define MAX_MARCHING_STEPS 50
#define EPSILON 0.001


struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		// position stored as 4 floats (4th component is supposed to be 1.0)
		Eigen::Vector4f position;
};

int main()
{
	// Parameters for the image plane and viewport
	const auto aspectRatio = 1.0;
	const unsigned int imageWidth = 50;
	const unsigned int imageHeight = static_cast<int>(imageWidth / aspectRatio);
	const auto viewportHeight = 2.0;
	const auto viewportWidth = aspectRatio * viewportHeight;

	// Camera parameters
	const Eigen::Vector3f cameraOrigin(0.0, 0.0, 0.0);
	const Eigen::Vector3f focalLength(0.0, 0.0, 1.0);

	// Spatial points
	const Eigen::Vector3f horizontal(viewportWidth, 0.0, 0.0);
	const Eigen::Vector3f vertical(0.0, viewportHeight, 0.0);
	const Eigen::Vector3f lowerLeftCorner = cameraOrigin - horizontal / 2.0 - vertical / 2.0 - focalLength;

	// Transform from TSDF space to camera space (only translation)
	Eigen::Vector3f translation(-1.0, 0.0, -1.0);

	// Init implicit surface
	Torus implicitTorus = Torus(Eigen::Vector3d(0.5, 0.5, 0.5), 0.4, 0.1);

	// Fill spatial grid with distance to the implicit surface
	unsigned int mc_res = 50;
	Volume vol(Eigen::Vector3d(-0.1, -0.1, -0.1), Eigen::Vector3d(1.1, 1.1, 1.1), mc_res, mc_res, mc_res, 1);
	for (unsigned int x = 0; x < vol.getDimX(); x++)
	{
		for (unsigned int y = 0; y < vol.getDimY(); y++)
		{
			for (unsigned int z = 0; z < vol.getDimZ(); z++)
			{
				Eigen::Vector3d p = vol.pos(x, y, z);
				double val = implicitTorus.Eval(p);
				if (val < TRUNCATION)
					vol.set(x, y, z, val);
				else
					vol.set(x, y, z, TRUNCATION);
			}
		}
	}

	// Traverse the image pixel by pixel
	for (unsigned int j = imageHeight - 1; j >= 0; --j)
	{
		for (unsigned int i = 0; i < imageWidth; ++i)
		{
			// Fractional steps along viewport in horizontal and vertical directions
			const auto u = double(i) / (imageWidth - 1);
			const auto v = double(j) / (imageHeight - 1);

			const Eigen::Vector3f rayDirection = lowerLeftCorner + u * horizontal + v * vertical - cameraOrigin;

			// Implicit cast to int, Volume::get expects Vector3i
			// double start = vol.get(0, 0, 0);
			double start = 1.0;
			double step = start; 

			for (unsigned int s = 0; s < MAX_MARCHING_STEPS; ++s)
			{
				Eigen::Vector3f p = cameraOrigin + step * rayDirection;
				// Converting camera space to grid TSDF space
				Eigen::Vector3f pTrans = p - translation;
				// Think carefully if this cast is correct or not
				if (!vol.outOfVolume(int(pTrans[0]), int(pTrans[1]), int(pTrans[2])))
				{
					double dist = vol.get(pTrans.cast<int>());
					if (dist < EPSILON)
					{
						std::cout << "INTERSECTION FOUND!" << std::endl;
						break;
					}
					step += dist;
					std::cout << dist << std::endl;
				}
				else
					break;
			}
		}
	}

	return 0;
}