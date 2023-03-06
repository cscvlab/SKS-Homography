/*============================================================================
Copyright 2017 Toby Collins
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "HO.hpp"

// Copy OpenCV's namespace HomographyHO, Here change its name to namespace cv::my_ho
// https://github.com/opencv/opencv/blob/17234f82d025e3bbfbf611089637e5aa2038e7b8/modules/calib3d/src/ippe.cpp line 866

using namespace cv;

namespace my_ho {
    void my_normalizeDataIsotropic(InputArray _Data, OutputArray _DataN, OutputArray _T, OutputArray _Ti)
    {
        Mat Data = _Data.getMat();
        int numPoints = Data.rows * Data.cols;
        CV_Assert(Data.rows == 1 || Data.cols == 1);
        CV_Assert(Data.channels() == 2 || Data.channels() == 3);
        CV_Assert(numPoints >= 4);

        int dataType = _Data.type();
        CV_CheckType(dataType, dataType == CV_32FC2 || dataType == CV_32FC3 || dataType == CV_64FC2 || dataType == CV_64FC3,
            "Type of _Data must be one of CV_32FC2, CV_32FC3, CV_64FC2, CV_64FC3");

        _DataN.create(2, numPoints, CV_64FC1);

        _T.create(3, 3, CV_64FC1);
        _Ti.create(3, 3, CV_64FC1);

        Mat DataN = _DataN.getMat();
        Mat T = _T.getMat();
        Mat Ti = _Ti.getMat();

        _T.setTo(0);
        _Ti.setTo(0);

        int numChannels = Data.channels();
        double xm = 0;
        double ym = 0;

        for (int i = 0; i < numPoints; i++)
        {
            if (numChannels == 2)
            {
                if (dataType == CV_32FC2)
                {
                    xm = xm + Data.at<Vec2f>(i)[0];
                    ym = ym + Data.at<Vec2f>(i)[1];
                }
                else
                {
                    xm = xm + Data.at<Vec2d>(i)[0];
                    ym = ym + Data.at<Vec2d>(i)[1];
                }
            }
            else
            {
                if (dataType == CV_32FC3)
                {
                    xm = xm + Data.at<Vec3f>(i)[0];
                    ym = ym + Data.at<Vec3f>(i)[1];
                }
                else
                {
                    xm = xm + Data.at<Vec3d>(i)[0];
                    ym = ym + Data.at<Vec3d>(i)[1];
                }
            }
        }
        xm = xm / static_cast<double>(numPoints);
        ym = ym / static_cast<double>(numPoints);

        double kappa = 0;
        double xh, yh;


        for (int i = 0; i < numPoints; i++)
        {

            if (numChannels == 2)
            {
                if (dataType == CV_32FC2)
                {
                    xh = Data.at<Vec2f>(i)[0] - xm;
                    yh = Data.at<Vec2f>(i)[1] - ym;
                }
                else
                {
                    xh = Data.at<Vec2d>(i)[0] - xm;
                    yh = Data.at<Vec2d>(i)[1] - ym;
                }
            }
            else
            {
                if (dataType == CV_32FC3)
                {
                    xh = Data.at<Vec3f>(i)[0] - xm;
                    yh = Data.at<Vec3f>(i)[1] - ym;
                }
                else
                {
                    xh = Data.at<Vec3d>(i)[0] - xm;
                    yh = Data.at<Vec3d>(i)[1] - ym;
                }
            }

            DataN.at<double>(0, i) = xh;
            DataN.at<double>(1, i) = yh;
            kappa = kappa + xh * xh + yh * yh;
        }
        double beta = sqrt(2 * numPoints / kappa);
        DataN = DataN * beta;

        T.at<double>(0, 0) = 1.0 / beta;
        T.at<double>(1, 1) = 1.0 / beta;

        T.at<double>(0, 2) = xm;
        T.at<double>(1, 2) = ym;

        T.at<double>(2, 2) = 1;

        Ti.at<double>(0, 0) = beta;
        Ti.at<double>(1, 1) = beta;

        Ti.at<double>(0, 2) = -beta * xm;
        Ti.at<double>(1, 2) = -beta * ym;

        Ti.at<double>(2, 2) = 1;
    }

    void my_homographyHO(InputArray _srcPoints, InputArray _targPoints, Matx33d& H)
    {
        Mat DataA, DataB, TA, TAi, TB, TBi;

        my_normalizeDataIsotropic(_srcPoints, DataA, TA, TAi);
        my_normalizeDataIsotropic(_targPoints, DataB, TB, TBi);

        int n = DataA.cols;
        CV_Assert(n == DataB.cols);

        Mat C1(1, n, CV_64FC1);
        Mat C2(1, n, CV_64FC1);
        Mat C3(1, n, CV_64FC1);
        Mat C4(1, n, CV_64FC1);

        double mC1 = 0, mC2 = 0, mC3 = 0, mC4 = 0;

        for (int i = 0; i < n; i++)
        {
            C1.at<double>(0, i) = -DataB.at<double>(0, i) * DataA.at<double>(0, i);
            C2.at<double>(0, i) = -DataB.at<double>(0, i) * DataA.at<double>(1, i);
            C3.at<double>(0, i) = -DataB.at<double>(1, i) * DataA.at<double>(0, i);
            C4.at<double>(0, i) = -DataB.at<double>(1, i) * DataA.at<double>(1, i);

            mC1 += C1.at<double>(0, i);
            mC2 += C2.at<double>(0, i);
            mC3 += C3.at<double>(0, i);
            mC4 += C4.at<double>(0, i);
        }

        mC1 /= n;
        mC2 /= n;
        mC3 /= n;
        mC4 /= n;

        Mat Mx(n, 3, CV_64FC1);
        Mat My(n, 3, CV_64FC1);


        for (int i = 0; i < n; i++)
        {
            Mx.at<double>(i, 0) = C1.at<double>(0, i) - mC1;
            Mx.at<double>(i, 1) = C2.at<double>(0, i) - mC2;
            Mx.at<double>(i, 2) = -DataB.at<double>(0, i);

            My.at<double>(i, 0) = C3.at<double>(0, i) - mC3;
            My.at<double>(i, 1) = C4.at<double>(0, i) - mC4;
            My.at<double>(i, 2) = -DataB.at<double>(1, i);
        }

        Mat DataAT, DataADataAT;

        transpose(DataA, DataAT);
        DataADataAT = DataA * DataAT;
        double dt = DataADataAT.at<double>(0, 0) * DataADataAT.at<double>(1, 1) - DataADataAT.at<double>(0, 1) * DataADataAT.at<double>(1, 0);

        Mat DataADataATi(2, 2, CV_64FC1);
        DataADataATi.at<double>(0, 0) = DataADataAT.at<double>(1, 1) / dt;
        DataADataATi.at<double>(0, 1) = -DataADataAT.at<double>(0, 1) / dt;
        DataADataATi.at<double>(1, 0) = -DataADataAT.at<double>(1, 0) / dt;
        DataADataATi.at<double>(1, 1) = DataADataAT.at<double>(0, 0) / dt;

        Mat Pp = DataADataATi * DataA;

        Mat Bx = Pp * Mx;
        Mat By = Pp * My;

        Mat Ex = DataAT * Bx;
        Mat Ey = DataAT * By;

        Mat D(2 * n, 3, CV_64FC1);


        for (int i = 0; i < n; i++)
        {
            D.at<double>(i, 0) = Mx.at<double>(i, 0) - Ex.at<double>(i, 0);
            D.at<double>(i, 1) = Mx.at<double>(i, 1) - Ex.at<double>(i, 1);
            D.at<double>(i, 2) = Mx.at<double>(i, 2) - Ex.at<double>(i, 2);

            D.at<double>(i + n, 0) = My.at<double>(i, 0) - Ey.at<double>(i, 0);
            D.at<double>(i + n, 1) = My.at<double>(i, 1) - Ey.at<double>(i, 1);
            D.at<double>(i + n, 2) = My.at<double>(i, 2) - Ey.at<double>(i, 2);
        }

        Mat DT, DDT;
        transpose(D, DT);
        DDT = DT * D;

        Mat S, U;
        eigen(DDT, S, U);

        Mat h789(3, 1, CV_64FC1);
        h789.at<double>(0, 0) = U.at<double>(2, 0);
        h789.at<double>(1, 0) = U.at<double>(2, 1);
        h789.at<double>(2, 0) = U.at<double>(2, 2);

        Mat h12 = -Bx * h789;
        Mat h45 = -By * h789;

        double h3 = -(mC1 * h789.at<double>(0, 0) + mC2 * h789.at<double>(1, 0));
        double h6 = -(mC3 * h789.at<double>(0, 0) + mC4 * h789.at<double>(1, 0));

        H(0, 0) = h12.at<double>(0, 0);
        H(0, 1) = h12.at<double>(1, 0);
        H(0, 2) = h3;

        H(1, 0) = h45.at<double>(0, 0);
        H(1, 1) = h45.at<double>(1, 0);
        H(1, 2) = h6;

        H(2, 0) = h789.at<double>(0, 0);
        H(2, 1) = h789.at<double>(1, 0);
        H(2, 2) = h789.at<double>(2, 0);

        H = Mat(TB * H * TAi);
        double h22_inv = 1 / H(2, 2);
        H = H * h22_inv;
    }
}