/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "DLT.hpp"

using namespace cv;

// Copy OpenCV's function runKernel. Here change its name to runKernel_DLT.
// https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/fundam.cpp line 118

// Total cost: >=27400 flops. N is the number of points.
namespace cv
{
    int runKernel_DLT(InputArray _m1, InputArray _m2, OutputArray _model)
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        int i, count = m1.checkVector(2);
        const Point2f* M = m1.ptr<Point2f>();
        const Point2f* m = m2.ptr<Point2f>();

        double LtL[9][9], W[9][1], V[9][9];
        Mat _LtL(9, 9, CV_64F, &LtL[0][0]);
        Mat matW(9, 1, CV_64F, W);
        Mat matV(9, 9, CV_64F, V);
        Mat _H0(3, 3, CV_64F, V[8]);
        Mat _Htemp(3, 3, CV_64F, V[7]);
        Point2f cM(0, 0), cm(0, 0), sM(0, 0), sm(0, 0);

        for (i = 0; i < count; i++)  // 4N
        {
            cm.x += m[i].x; cm.y += m[i].y;
            cM.x += M[i].x; cM.y += M[i].y;
        }
        // 16 flops
        cm.x /= count;
        cm.y /= count;
        cM.x /= count;
        cM.y /= count;

        for (i = 0; i < count; i++)  // 4N*3=12N
        {
            sm.x += fabs(m[i].x - cm.x);
            sm.y += fabs(m[i].y - cm.y);
            sM.x += fabs(M[i].x - cM.x);
            sM.y += fabs(M[i].y - cM.y);
        }

        if (fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
            fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON)
            return 0;
        // 16 flops
        sm.x = count / sm.x; sm.y = count / sm.y;
        sM.x = count / sM.x; sM.y = count / sM.y;

        double invHnorm[9] = { 1. / sm.x, 0, cm.x, 0, 1. / sm.y, cm.y, 0, 0, 1 };  // 8 flops
        double Hnorm2[9] = { sM.x, 0, -cM.x * sM.x, 0, sM.y, -cM.y * sM.y, 0, 0, 1 };  // 4 flops
        Mat _invHnorm(3, 3, CV_64FC1, invHnorm);
        Mat _Hnorm2(3, 3, CV_64FC1, Hnorm2);

        _LtL.setTo(Scalar::all(0));
        for (i = 0; i < count; i++)  // N*(4*2+10+45*4) = 198N
        {
            double x = (m[i].x - cm.x) * sm.x, y = (m[i].y - cm.y) * sm.y;
            double X = (M[i].x - cM.x) * sM.x, Y = (M[i].y - cM.y) * sM.y;
            double Lx[] = { X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x };
            double Ly[] = { 0, 0, 0, X, Y, 1, -y * X, -y * Y, -y };
            int j, k;
            for (j = 0; j < 9; j++)
                for (k = j; k < 9; k++)
                    LtL[j][k] += Lx[j] * Lx[k] + Ly[j] * Ly[k];
        }
        completeSymm(_LtL);  // no flops but time-consuming

        eigen(_LtL, matW, matV);   // at least 36*9^3 = 26244 flops (roughly estimated)
        _Htemp = _invHnorm * _H0;    // 5*3*3=45 flops
        _H0 = _Htemp * _Hnorm2;      // 45 flops
        _H0.convertTo(_model, _H0.type(), 1. / _H0.at<double>(2, 2));   // 12 flops

        return 1;
    }
}
