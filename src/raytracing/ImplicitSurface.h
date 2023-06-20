#ifndef IMPLICIT_SURFACE_H
#define IMPLICIT_SURFACE_H

#include <Eigen/Dense>

class Torus
{
public:
	Torus(const Eigen::Vector3d& center, double radius, double a) : m_center(center), m_radius(radius), m_a(a)
	{
	}

	double Eval(const Eigen::Vector3d& _x)
	{
		Eigen::Vector3d v1;
		Eigen::Vector2d v2(_x[0] - m_center[0], _x[1] - m_center[1]);
		v1 = _x - m_center;
		return (v1.squaredNorm() + m_radius * m_radius - m_a * m_a) * (v1.squaredNorm() + m_radius * m_radius - m_a * m_a) - 4 * m_radius * m_radius * (v2.squaredNorm());
	}

private:
	Eigen::Vector3d m_center;
	double m_radius;
	double m_a;
};

#endif // !IMPLICIT_SURFACE_H
