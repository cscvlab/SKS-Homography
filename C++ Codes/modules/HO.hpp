#pragma once
#ifndef __OPENCV_HO_H__
#define __OPENCV_HO_H__
#include <opencv2/core.hpp>

namespace cv::my_ho
{
	void homographyHO(InputArray _srcPoints, InputArray _targPoints, Matx33d& H);
}

#endif
