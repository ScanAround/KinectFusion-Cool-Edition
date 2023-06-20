#include <iostream>

#include <Eigen/Dense>
#include "ImplicitSurface.h"
#include "Volume.h"

int main()
{
	Torus implicitTorus = Torus(Eigen::Vector3d(0.5, 0.5, 0.5), 0.4, 0.1);

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
				vol.set(x, y, z, val);
			}
		}
	}

	return 0;
}