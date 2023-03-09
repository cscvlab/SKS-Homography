#pragma once
#ifndef __OPENCV_GPT_H__
#define __OPENCV_GPT_H__
#include <opencv2/core.hpp>


namespace cv
{
	void runKernel_GPT(std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& tar, cv::Mat& _model);
}

#endif
