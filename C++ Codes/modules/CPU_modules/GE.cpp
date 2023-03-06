/*
  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
  By downloading, copying, installing or using the software you agree to this license.
  If you do not agree to this license, do not download, install,
  copy or use the software.
                          BSD 3-Clause License
 Copyright (C) 2014, Olexa Bilaniuk, Hamid Bazargani & Robert Laganiere, all rights reserved.
 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:
   * Redistribution's of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
   * Redistribution's in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * The name of the copyright holders may not be used to endorse or promote products
     derived from this software without specific prior written permission.
 This software is provided by the copyright holders and contributors "as is" and
 any express or implied warranties, including, but not limited to, the implied
 warranties of merchantability and fitness for a particular purpose are disclaimed.
 In no event shall the Intel Corporation or contributors be liable for any direct,
 indirect, incidental, special, exemplary, or consequential damages
 (including, but not limited to, procurement of substitute goods or services;
 loss of use, data, or profits; or business interruption) however caused
 and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.
*/

/**
 * Bilaniuk, Olexa, Hamid Bazargani, and Robert Laganiere. "Fast Target
 * Recognition on Mobile Devices: Revisiting Gaussian Elimination for the
 * Estimation of Planar Homographies." In Computer Vision and Pattern
 * Recognition Workshops (CVPRW), 2014 IEEE Conference on, pp. 119-125.
 * IEEE, 2014.
 */

#include "GE.hpp"

// Copy OpenCV's function hFuncRefC, Here change its name to runKernelGE_Optimized
// https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/rho.cpp  line 1797
namespace cv
{
	void runKernelGE_Optimized(float* src, float* tar, float* result)
	{
		float x0 = src[0];
		float y0 = src[1];
		float x1 = src[2];
		float y1 = src[3];
		float x2 = src[4];
		float y2 = src[5];
		float x3 = src[6];
		float y3 = src[7];
		float X0 = tar[0];
		float Y0 = tar[1];
		float X1 = tar[2];
		float Y1 = tar[3];
		float X2 = tar[4];
		float Y2 = tar[5];
		float X3 = tar[6];
		float Y3 = tar[7];

		float x0X0 = x0 * X0, x1X1 = x1 * X1, x2X2 = x2 * X2, x3X3 = x3 * X3;
		float x0Y0 = x0 * Y0, x1Y1 = x1 * Y1, x2Y2 = x2 * Y2, x3Y3 = x3 * Y3;
		float y0X0 = y0 * X0, y1X1 = y1 * X1, y2X2 = y2 * X2, y3X3 = y3 * X3;
		float y0Y0 = y0 * Y0, y1Y1 = y1 * Y1, y2Y2 = y2 * Y2, y3Y3 = y3 * Y3;

		float minor[2][4] = { { x0 - x2, x1 - x2, x2, x3 - x2 },
		{ y0 - y2, y1 - y2, y2, y3 - y2 } };
		float major[3][8] = { { x2X2 - x0X0, x2X2 - x1X1, -x2X2, x2X2 - x3X3, x2Y2 - x0Y0, x2Y2 - x1Y1, -x2Y2, x2Y2 - x3Y3 },
		{ y2X2 - y0X0, y2X2 - y1X1, -y2X2, y2X2 - y3X3, y2Y2 - y0Y0, y2Y2 - y1Y1, -y2Y2, y2Y2 - y3Y3 },
		{ (X0 - X2), (X1 - X2), (X2), (X3 - X2), (Y0 - Y2), (Y1 - Y2), (Y2), (Y3 - Y2) } };

		float scalar1 = minor[0][0], scalar2 = minor[0][1];
		minor[1][1] = minor[1][1] * scalar1 - minor[1][0] * scalar2;

		major[0][1] = major[0][1] * scalar1 - major[0][0] * scalar2;
		major[1][1] = major[1][1] * scalar1 - major[1][0] * scalar2;
		major[2][1] = major[2][1] * scalar1 - major[2][0] * scalar2;

		major[0][5] = major[0][5] * scalar1 - major[0][4] * scalar2;
		major[1][5] = major[1][5] * scalar1 - major[1][4] * scalar2;
		major[2][5] = major[2][5] * scalar1 - major[2][4] * scalar2;

		scalar2 = minor[0][3];
		minor[1][3] = minor[1][3] * scalar1 - minor[1][0] * scalar2;

		major[0][3] = major[0][3] * scalar1 - major[0][0] * scalar2;
		major[1][3] = major[1][3] * scalar1 - major[1][0] * scalar2;
		major[2][3] = major[2][3] * scalar1 - major[2][0] * scalar2;

		major[0][7] = major[0][7] * scalar1 - major[0][4] * scalar2;
		major[1][7] = major[1][7] * scalar1 - major[1][4] * scalar2;
		major[2][7] = major[2][7] * scalar1 - major[2][4] * scalar2;

		scalar1 = minor[1][1]; scalar2 = minor[1][3];
		major[0][3] = major[0][3] * scalar1 - major[0][1] * scalar2;
		major[1][3] = major[1][3] * scalar1 - major[1][1] * scalar2;
		major[2][3] = major[2][3] * scalar1 - major[2][1] * scalar2;

		major[0][7] = major[0][7] * scalar1 - major[0][5] * scalar2;
		major[1][7] = major[1][7] * scalar1 - major[1][5] * scalar2;
		major[2][7] = major[2][7] * scalar1 - major[2][5] * scalar2;

		scalar2 = minor[1][0];
		//minor[0][0] = minor[0][0] * scalar1 - minor[0][1] * scalar2;
		minor[0][0] = minor[0][0] * scalar1;
		major[0][0] = major[0][0] * scalar1 - major[0][1] * scalar2;
		major[1][0] = major[1][0] * scalar1 - major[1][1] * scalar2;
		major[2][0] = major[2][0] * scalar1 - major[2][1] * scalar2;

		major[0][4] = major[0][4] * scalar1 - major[0][5] * scalar2;
		major[1][4] = major[1][4] * scalar1 - major[1][5] * scalar2;
		major[2][4] = major[2][4] * scalar1 - major[2][5] * scalar2;

		scalar1 = 1.0f / minor[0][0];
		major[0][0] *= scalar1;
		major[1][0] *= scalar1;
		major[2][0] *= scalar1;
		major[0][4] *= scalar1;
		major[1][4] *= scalar1;
		major[2][4] *= scalar1;

		scalar1 = 1.0f / minor[1][1];
		major[0][1] *= scalar1;
		major[1][1] *= scalar1;
		major[2][1] *= scalar1;
		major[0][5] *= scalar1;
		major[1][5] *= scalar1;
		major[2][5] *= scalar1;


		scalar1 = minor[0][2]; scalar2 = minor[1][2];
		major[0][2] -= major[0][0] * scalar1 + major[0][1] * scalar2;
		major[1][2] -= major[1][0] * scalar1 + major[1][1] * scalar2;
		major[2][2] -= major[2][0] * scalar1 + major[2][1] * scalar2;

		major[0][6] -= major[0][4] * scalar1 + major[0][5] * scalar2;
		major[1][6] -= major[1][4] * scalar1 + major[1][5] * scalar2;
		major[2][6] -= major[2][4] * scalar1 + major[2][5] * scalar2;

		/* Only major matters now. R(3) and R(7) correspond to the hollowed-out rows. */
		scalar1 = major[0][7];
		major[1][7] /= scalar1;
		major[2][7] /= scalar1;

		scalar1 = major[0][0]; major[1][0] -= scalar1 * major[1][7]; major[2][0] -= scalar1 * major[2][7];
		scalar1 = major[0][1]; major[1][1] -= scalar1 * major[1][7]; major[2][1] -= scalar1 * major[2][7];
		scalar1 = major[0][2]; major[1][2] -= scalar1 * major[1][7]; major[2][2] -= scalar1 * major[2][7];
		scalar1 = major[0][3]; major[1][3] -= scalar1 * major[1][7]; major[2][3] -= scalar1 * major[2][7];
		scalar1 = major[0][4]; major[1][4] -= scalar1 * major[1][7]; major[2][4] -= scalar1 * major[2][7];
		scalar1 = major[0][5]; major[1][5] -= scalar1 * major[1][7]; major[2][5] -= scalar1 * major[2][7];
		scalar1 = major[0][6]; major[1][6] -= scalar1 * major[1][7]; major[2][6] -= scalar1 * major[2][7];


		/* One column left (Two in fact, but the last one is the homography) */
		scalar1 = major[1][3];

		major[2][3] /= scalar1;
		scalar1 = major[1][0]; major[2][0] -= scalar1 * major[2][3];
		scalar1 = major[1][1]; major[2][1] -= scalar1 * major[2][3];
		scalar1 = major[1][2]; major[2][2] -= scalar1 * major[2][3];
		scalar1 = major[1][4]; major[2][4] -= scalar1 * major[2][3];
		scalar1 = major[1][5]; major[2][5] -= scalar1 * major[2][3];
		scalar1 = major[1][6]; major[2][6] -= scalar1 * major[2][3];
		scalar1 = major[1][7]; major[2][7] -= scalar1 * major[2][3];
		
		result[0] = major[2][0];
		result[1] = major[2][1];
		result[2] = major[2][2];

		result[3] = major[2][4];
		result[4] = major[2][5];
		result[5] = major[2][6];

		result[6] = major[2][7];
		result[7] = major[2][3];
		result[8] = 1.0;
		
	}
}