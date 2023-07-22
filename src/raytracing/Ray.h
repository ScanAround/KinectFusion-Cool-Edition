#ifndef RAY_H
#define RAY_H

#include <Eigen/Dense>

class Ray
{
public:
	Ray(const Eigen::Vector3f _origin, const Eigen::Vector3f _dir);
	
	Eigen::Vector3f at(const double t) const;

	Eigen::Vector3f getOrigin() const;

	Eigen::Vector3f getDirection() const;

private:
	const Eigen::Vector3f origin;
	const Eigen::Vector3f dir;
};


#endif  // RAY_H