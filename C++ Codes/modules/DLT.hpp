#pragma once
#ifndef __OPENCV_DLT_H__
#define __OPENCV_DLT_H__
#include <opencv2/core.hpp>
namespace cv
{
    int runKernel_DLT(cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model);
}

#endif
