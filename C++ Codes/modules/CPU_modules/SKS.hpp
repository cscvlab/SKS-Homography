#pragma once
#ifndef __OPENCV_SKS_H__
#define __OPENCV_SKS_H__

#include "opencv2/core.hpp"


namespace cv::sks {

	/**
	* Input the correct corresponding points, for example, the SIFT keypoint.
	*
	* SKS_homography_decompostion_computation
	* @return the homography matrix.
	*/

    int runKernel_ACA(float* src, float* tar, float* result);
    int runKernel_ACA_double(float* src, float* tar, double* result);
    int runKernel_SKS(float* src, float* tar, float* result);
    int runKernel_SKS_double(float* src, float* tar, double* result);

}

#endif
