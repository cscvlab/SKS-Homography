/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "GPT.hpp"

// Copy OpenCV's function getPerspectiveTransform, Here change its name to runKernel_GPT
// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/imgwarp.cpp  line 3407
// 1900+32 flops roughly estimated
namespace cv
{
    void runKernel_GPT(std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& tar, cv::Mat& _model) {
        Mat M(3, 3, CV_64F), X(8, 1, CV_64F, M.ptr());
        double a[8][8], b[8];
        Mat A(8, 8, CV_64F, a), B(8, 1, CV_64F, b);
        // flops = 32
        for (int i = 0; i < 4; ++i) {
            a[i][0] = a[i + 4][3] = src[i].x;
            a[i][1] = a[i + 4][4] = src[i].y;
            a[i][2] = a[i + 4][5] = 1;
            a[i][3] = a[i][4] = a[i][5] =
                a[i + 4][0] = a[i + 4][1] = a[i + 4][2] = 0;
            a[i][6] = -src[i].x * tar[i].x;
            a[i][7] = -src[i].y * tar[i].x;
            a[i + 4][6] = -src[i].x * tar[i].y;
            a[i + 4][7] = -src[i].y * tar[i].y;
            b[i] = tar[i].x;
            b[i + 4] = tar[i].y;
        }
        solve(A, B, X);  // flops = 1900 (roughly estimated)
        M.ptr<double>()[8] = 1.;
        M.copyTo(_model);
    }
}
