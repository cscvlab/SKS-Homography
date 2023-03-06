#pragma once
#ifndef __OPENCV_HO_H__
#define __OPENCV_HO_H__
#include <opencv2/core.hpp>

namespace my_ho
{
	void my_homographyHO(cv::InputArray _srcPoints, cv::InputArray _targPoints, cv::Matx33d& H);
}

#endif
