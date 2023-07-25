#pragma once

#ifndef VOLUME_H
#define VOLUME_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <limits>
#include <Eigen/Dense>
typedef unsigned int uint;

//! A regular volume dataset
class Volume
{
public:

	//! Initializes an empty volume dataset.
	Volume(Eigen::Vector3d min_, Eigen::Vector3d max_, uint dx_ = 10, uint dy_ = 10, uint dz_ = 10, uint dim = 1);

	Volume(const Volume& v);

	~Volume();

	inline void computeMinMaxValues(double& minVal, double& maxVal) const
	{
		minVal = std::numeric_limits<double>::max();
		maxVal = -minVal;
		for (uint i1 = 0; i1 < dx*dy*dz; i1++)
		{
			if (minVal > vol[i1]) minVal = vol[i1];
			if (maxVal < vol[i1]) maxVal = vol[i1];
		}
	}

	//! Computes spacing in x,y,z-directions.
	void compute_ddx_dddx();

	//! Zeros out the memory
	void zeroOutMemory();

	// check index out of volume 
	inline bool outOfVolume(uint _x, uint _y, uint _z)
	{
		if (_x < 0 || _x > dx - 1 || _y < 0 || _y > dy - 1 || _z < 0 || _z > dz - 1)
			return true;
		return false;
	}

	inline Eigen::Vector3f worldToGrid(const Eigen::Vector3f& p)
	{
		Eigen::Vector3f coord(0.0, 0.0, 0.0);

		coord[0] = (p[0] - min[0]) / (max[0] - min[0]) / ddx;
		coord[1] = (p[1] - min[1]) / (max[1] - min[1]) / ddy;
		coord[2] = (p[2] - min[2]) / (max[2] - min[2]) / ddz;

		return coord;
	}

	//! Set the value at i.
	inline void set(uint i, double val)
	{
		if (val > maxValue)
			maxValue = val;

		if (val < minValue)
			minValue = val;

		vol[i] = val;
	}

	//! Set the value at (x_, y_, z_).
	inline void set(uint x_, uint y_, uint z_, double val)
	{
		vol[getPosFromTuple(x_, y_, z_)] = val;
	};

	//! Get the value at (x_, y_, z_).
	inline double get(uint i) const
	{
		return vol[i];
	};

	//! Get the value at (x_, y_, z_).
	inline double get(uint x_, uint y_, uint z_) const
	{
		return vol[getPosFromTuple(x_, y_, z_)];
	};

	//! Get the value at (pos.x, pos.y, pos.z).
	inline double get(const Eigen::Vector3i& pos_) const
	{
		return(get(pos_[0], pos_[1], pos_[2]));
	}

	//! Returns the cartesian x-coordinates of node (i,..).
	inline double posX(int i) const
	{
		return min[0] + diag[0] * (double(i)*ddx);
	}

	//! Returns the cartesian y-coordinates of node (..,i,..).
	inline double posY(int i) const
	{
		return min[1] + diag[1] * (double(i)*ddy);
	}

	//! Returns the cartesian z-coordinates of node (..,i).
	inline double posZ(int i) const
	{
		return min[2] + diag[2] * (double(i)*ddz);
	}

	//! Returns the cartesian coordinates of node (i,j,k).
	inline Eigen::Vector3d pos(int i, int j, int k) const
	{
		Eigen::Vector3d coord(0, 0, 0);

		coord[0] = min[0] + (max[0] - min[0])*(double(i)*ddx);
		coord[1] = min[1] + (max[1] - min[1])*(double(j)*ddy);
		coord[2] = min[2] + (max[2] - min[2])*(double(k)*ddz);

		return coord;
	}

	//! Returns the Data.
	double* getData();

	//! Sets all entries in the volume to '0'
	void clean();

	//! Returns number of cells in x-dir.
	inline uint getDimX() const { return dx; }

	//! Returns number of cells in y-dir.
	inline uint getDimY() const { return dy; }

	//! Returns number of cells in z-dir.
	inline uint getDimZ() const { return dz; }

	inline Eigen::Vector3d getMin() { return min; }
	inline Eigen::Vector3d getMax() { return max; }

	//! Sets minimum extension
	void SetMin(Eigen::Vector3d min_);

	//! Sets maximum extension
	void SetMax(Eigen::Vector3d max_);

	inline uint getPosFromTuple(int x, int y, int z) const
	{
		return x*dy*dz + y*dz + z;
	}

	void writePointCloud(const std::string& filename)
	{
		std::ofstream file(filename);
		file << "OFF" << std::endl;
		std::vector<std::vector<unsigned int>> vertices;
		for (unsigned int x = 0; x < getDimX(); x++)
		{
			for (unsigned int y = 0; y < getDimY(); y++)
			{
				for (unsigned int z = 0; z < getDimZ(); z++)
				{
					if (abs(get(x, y, z)) <= 0.05)
					{	
						std::vector<unsigned int> v = {x, y, z};
						vertices.push_back(v);
					}	
				}
			}
		}
		file << vertices.size() << " 0 0" << std::endl;
		for (auto& elem: vertices)
		{
			file << elem[0] << " " << elem[1] << " " << elem[2] << std::endl;
		}

	}


	//! Lower left and Upper right corner.
	Eigen::Vector3d min, max;

	//! max-min
	Eigen::Vector3d diag;

	double ddx, ddy, ddz;
	double dddx, dddy, dddz;

	//! Number of cells in x, y and z-direction.
	uint dx, dy, dz;

	double* vol;

	double maxValue, minValue;

	uint m_dim;

private:

	//! x,y,z access to vol*
	inline double vol_access(int x, int y, int z) const
	{
		return vol[getPosFromTuple(x, y, z)];
	}
};

#endif // VOLUME_H
