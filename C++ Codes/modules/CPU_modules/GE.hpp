#pragma once
#ifndef __OPENCV_GE_H__
#define __OPENCV_GE_H__

#include "opencv2/core.hpp"

namespace cv
{
	void runKernelGE_Optimized(float* src, float* tar, float* result);
}

#endif
