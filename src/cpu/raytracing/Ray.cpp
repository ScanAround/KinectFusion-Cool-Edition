#include "Ray.h"

Ray::Ray(const Eigen::Vector3f _origin, const Eigen::Vector3f _dir): origin(_origin), dir(_dir)
{

}

Eigen::Vector3f Ray::at(const double t) const
{
    return origin + t*dir;
}

Eigen::Vector3f Ray::getOrigin() const 
{
    return origin;
}

Eigen::Vector3f Ray::getDirection() const 
{
    return dir;
}