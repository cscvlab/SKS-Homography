/**
A C++ code of Cai's homography solver SKS [1] and ACA [1].
This software will only be used for non-commercial research purposes. See License.md.
For commercial use, please contact the authors.

    Similarity-Kernel-Similarity (SKS) Decomposition of 2D homographies:

              H = H_S2^(-1) * H_K * H_S1 = H_S2^(-1) * H_E * H_T2_inv * H_G * H_T1 * H_E * H_S1

  FLOPs: 169 = 157 (compute homographies up to a scale from 4 point correspondences) + 12 (normalization)

  Detailed explanation goes here
  Inputs: SourcePts --- source points;
              TargetPts --- target points;
              Both of them (3*4 matrices) are represented in homogeneous coordinates with the last elements 1.
  Outputs: H --- homography between source and target planes  with normalization of the last element;


    Affine-Core-Affine (ACA) Decomposition of 2D homographies:

              H = H_A2^(-1) * H_C * H_A1

  FLOPs: 97 = 85 (compute homographies up to a scale from 4 point correspondences) + 12 (normalization)

  Detailed explanation goes here
  Inputs: SourcePts --- source points;
              TargetPts --- target points;
              Both of them (3*4 matrices) are represented in homogeneous coordinates.
  Outputs: H --- homography between source and target planes  with normalization of the last element;

 REFERENCE:
   [1] S. Cai, et al., "Fast and Interpretable 2D Homography Decomposition: Similarity-Kernel-Similarity (SKS)
        and Affine-Core-Affine (ACA)", submitted.
*/

#include "SKS.h"

namespace cv::sks {

/*
    Affine-Core-Affine (ACA) Decomposition of 2D homography:
           H = H_A2^(-1) * H_C * H_A1
    FLOPs: 97 = 85 (compute homographies up to a scale) +  12 (normalization)
    Inputs: src-- source points;  (a vector containing four 2D coordinates)
            tar --- target points (a vector containing four 2D coordinates);
    Outputs: result --- homography between source and target planes with normalization of the last element.
*/

	int runKernel_ACA(float* src, float* tar, float* result)
	{

		// compute the affine trans. H_A1 and other variables on source plane, 15 flops
		float M1N1_x = src[2] - src[0], M1P1_x = src[4] - src[0], M1Q1_x = src[6] - src[0];	// 3 
		float M1N1_y = src[3] - src[1], M1P1_y = src[5] - src[1], M1Q1_y = src[7] - src[1];	// 3
		float fA1 = M1N1_x * M1P1_y - M1N1_y * M1P1_x;	// 3
		float Q3_x = M1P1_y * M1Q1_x - M1P1_x * M1Q1_y;	 // 3
		float Q3_y = M1N1_x * M1Q1_y - M1N1_y * M1Q1_x;	 // 3
        //        [  M1P1_y   -M1P1_x          ]   [1        -src[0] ]
        // H_A1 = [ -M1N1_y    M1N1_x          ] * [     1   -src[1] ]
        //        [                       f_A1 ]   [             1   ]

		// compute H_A2 and other variables on target plane, 15 flops
		float M2N2_x = tar[2] - tar[0], M2P2_x = tar[4] - tar[0], M2Q2_x = tar[6] - tar[0];	// 3
		float M2N2_y = tar[3] - tar[1], M2P2_y = tar[5] - tar[1], M2Q2_y = tar[7] - tar[1];	// 3
		float fA2 = M2N2_x * M2P2_y - M2N2_y * M2P2_x;	// 3
		float Q4_x = M2P2_y * M2Q2_x - M2P2_x * M2Q2_y;	// 3
		float Q4_y = M2N2_x * M2Q2_y - M2N2_y * M2Q2_x;	// 3
        //            [  M2N2_x   M2P2_x    tar[0] ]
        // H_A2_inv = [  M2N2_y   M2P2_y    tar[1] ]
        //            [                        1   ]


		// obtain the core transformation H_C, 12 flops
		float tt1 = fA1 - Q3_x - Q3_y;	// 2
		float C11 = Q3_y * Q4_x * tt1;	// 2
		float C22 = Q3_x * Q4_y * tt1;	// 2
		float C33 = Q3_x * Q3_y * (fA2 - Q4_x - Q4_y);	// 4
		float C31 = C11 - C33;	// 1	
		float C32 = C22 - C33;	// 1
        //       [   C11          0       0  ]
        // H_C = [    0          C22      0  ]
        //       [ C11-C33     C22-C33   C33 ]


		// obtain some intermediate variables, 10 flops
		float tt3 = tar[0] * C33;     // 1
		float tt4 = tar[1] * C33;	 // 1
		float H1_11 = tar[2] * C11 - tt3;	// 2
		float H1_12 = tar[4] * C22 - tt3;	// 2
		float H1_21 = tar[3] * C11 - tt4;	// 2
		float H1_22 = tar[5] * C22 - tt4;	// 2
        // H1 = H_A2_inv * H_C
        // only compute the 2*2 upper-left elements. C_33*M2 are repeated twice. no need to compute the last row
        //      [ C11*tar[2]-C33*tar[0]     C22*tar[4]-C33*tar[0]     C33*tar[0]  ]
        // H1 = [ C11*tar[3]-C33*tar[1]     C22*tar[5]-C33*tar[1]     C33*tar[1]  ]
        //      [          C31                       C32                 C33      ]

		// obtain H, 33 flops
		result[0] = H1_11 * M1P1_y - H1_12 * M1N1_y;
		result[1] = H1_12 * M1N1_x - H1_11 * M1P1_x;
		result[3] = H1_21 * M1P1_y - H1_22 * M1N1_y;
		result[4] = H1_22 * M1N1_x - H1_21 * M1P1_x;
		result[6] = C31 * M1P1_y - C32 * M1N1_y;
		result[7] = C32 * M1N1_x - C31 * M1P1_x;
		result[2] = tt3 * fA1 - result[0] * src[0] - result[1] * src[1];
		result[5] = tt4 * fA1 - result[3] * src[0] - result[4] * src[1];
		result[8] = C33 * fA1 - result[6] * src[0] - result[7] * src[1];
        // firstly compute the first two columns of H = H1 * H_A1
        //                 [ H1_11*M1P1_y-H1_12*M1N1_y     H1_12*M1N1_x-H1_11*M1P1_x ]
        // H2 = H(:,1:2) = [ H1_21*M1P1_y-H1_22*M1N1_y     H1_22*M1N1_x-H1_21*M1P1_x ]
        //                 [   C31*M1P1_y-C32*M1N1_y          C32*M1N1_x-C31*M1P1_x  ]
        // then compute the last column of H utilizing the computed H2
        //            [ f_A1*C33*tar[0]-H2_11*src[0]-H2_12*src[1]  ]
        // H(:,3) =   [ f_A1*C33*tar[1]-H2_21*src[0]-H2_22*src[1]  ]
        //            [     f_A1*C33-H2_31*src[0]-H2_32*src[1]     ]


        // normalization based on the last element of H, 12 flops
		float result8_ = 1 / result[8];
		for (int i = 0; i < 8; i++) {
			result[i] *= result8_;
		}
		result[8] = 1.0f;


		return 0;
	}

	int runKernel_ACA_double(float* src, float* tar, double* result)
	{

		// compute the affine trans. H_A1 and other variables on source plane, 15 flops
		double M1N1_x = src[2] - src[0], M1P1_x = src[4] - src[0], M1Q1_x = src[6] - src[0];	// 3 
		double M1N1_y = src[3] - src[1], M1P1_y = src[5] - src[1], M1Q1_y = src[7] - src[1];	// 3
		double fA1 = M1N1_x * M1P1_y - M1N1_y * M1P1_x;	// 3
		double Q3_x = M1P1_y * M1Q1_x - M1P1_x * M1Q1_y;	 // 3
		double Q3_y = M1N1_x * M1Q1_y - M1N1_y * M1Q1_x;	 // 3
        //        [  M1P1_y   -M1P1_x          ]   [1        -src[0] ]
        // H_A1 = [ -M1N1_y    M1N1_x          ] * [     1   -src[1] ]
        //        [                       f_A1 ]   [             1   ]

		// compute H_A2 and other variables on target plane, 15 flops
		double M2N2_x = tar[2] - tar[0], M2P2_x = tar[4] - tar[0], M2Q2_x = tar[6] - tar[0];	// 3
		double M2N2_y = tar[3] - tar[1], M2P2_y = tar[5] - tar[1], M2Q2_y = tar[7] - tar[1];	// 3
		double fA2 = M2N2_x * M2P2_y - M2N2_y * M2P2_x;	// 3
		double Q4_x = M2P2_y * M2Q2_x - M2P2_x * M2Q2_y;	// 3
		double Q4_y = M2N2_x * M2Q2_y - M2N2_y * M2Q2_x;	// 3
        //            [  M2N2_x   M2P2_x    tar[0] ]
        // H_A2_inv = [  M2N2_y   M2P2_y    tar[1] ]
        //            [                        1   ]

		// obtain the core transformation H_C, 12 flops
		double tt1 = fA1 - Q3_x - Q3_y;	// 2
		double C11 = Q3_y * Q4_x * tt1;	// 2
		double C22 = Q3_x * Q4_y * tt1;	// 2
		double C33 = Q3_x * Q3_y * (fA2 - Q4_x - Q4_y);	// 4
		double C31 = C11 - C33;	// 1	
		double C32 = C22 - C33;	// 1
        //       [   C11          0       0  ]
        // H_C = [    0          C22      0  ]
        //       [ C11-C33     C22-C33   C33 ]

		// obtain some intermediate variables, 10 flops
		double tt3 = tar[0] * C33;     // 1
		double tt4 = tar[1] * C33;	 // 1
		double H1_11 = tar[2] * C11 - tt3;	// 2
		double H1_12 = tar[4] * C22 - tt3;	// 2
		double H1_21 = tar[3] * C11 - tt4;	// 2
		double H1_22 = tar[5] * C22 - tt4;	// 2
        // H1 = H_A2_inv * H_C
        // only compute the 2*2 upper-left elements. C_33*M2 are repeated twice. no need to compute the last row
        //      [ C11*tar[2]-C33*tar[0]     C22*tar[4]-C33*tar[0]     C33*tar[0]  ]
        // H1 = [ C11*tar[3]-C33*tar[1]     C22*tar[5]-C33*tar[1]     C33*tar[1]  ]
        //      [          C31                       C32                 C33      ]

		 // obtain H, 33 flops
		result[0] = H1_11 * M1P1_y - H1_12 * M1N1_y;
		result[1] = H1_12 * M1N1_x - H1_11 * M1P1_x;
		result[3] = H1_21 * M1P1_y - H1_22 * M1N1_y;
		result[4] = H1_22 * M1N1_x - H1_21 * M1P1_x;
		result[6] = C31 * M1P1_y - C32 * M1N1_y;
		result[7] = C32 * M1N1_x - C31 * M1P1_x;
		result[2] = tt3 * fA1 - result[0] * src[0] - result[1] * src[1];
		result[5] = tt4 * fA1 - result[3] * src[0] - result[4] * src[1];
		result[8] = C33 * fA1 - result[6] * src[0] - result[7] * src[1];
        // firstly compute the first two columns of H = H1 * H_A1
        //                 [ H1_11*M1P1_y-H1_12*M1N1_y     H1_12*M1N1_x-H1_11*M1P1_x ]
        // H2 = H(:,1:2) = [ H1_21*M1P1_y-H1_22*M1N1_y     H1_22*M1N1_x-H1_21*M1P1_x ]
        //                 [   C31*M1P1_y-C32*M1N1_y          C32*M1N1_x-C31*M1P1_x  ]
        // then compute the last column of H utilizing the computed H2
        //          [ f_A1*C33*tar[0]-H2_11*src[0]-H2_12*src[1]  ]
        // H(:,3) = [ f_A1*C33*tar[1]-H2_21*src[0]-H2_22*src[1]  ]
        //          [     f_A1*C33-H2_31*src[0]-H2_32*src[1]     ]

        // normalization based on the last element of H, 12 flops
		double result8_ = 1 / result[8];
		for (int i = 0; i < 8; i++) {
			result[i] *= result8_;
		}
		result[8] = 1.0f;


		return 0;
	}

/*
    Similarity-Kernel-Similarity (SKS) Decomposition:
               H = H_S2^(-1) * H_K * H_S1 = H_S2^(-1) * H_E * H_T2_inv * H_G * H_T1 * H_E * H_S1
    FLOPs: 169 = 157 (compute homographies up to a scale) + 12 (normalization)
    Inputs: src --- source points (a vector containing four 2D coordinates);
            tar --- target points (a vector containing four 2D coordinates);
    Outputs: result --- homography between source and target planes with normalization of the last element.
*/
	int runKernel_SKS(float* src, float* tar, float* result)
	{
        // compute the similarity transformation H_S1 on source plane based on two anchor points (e.g., M1 and N1), 9 flops
		float o1_x = 0.5 * (src[0] + src[2]);   // midpoint O1, 2 flops
		float o1_y = 0.5 * (src[1] + src[3]);   // 2
		float w1_x = o1_x - src[0];             // 1, vector O1N1 = M1O1
		float w1_y = src[1] - o1_y;             // 1
		float f_S1 = (w1_x * w1_x + w1_y * w1_y); // 3
        //        [  w1_x  w1_y    -w1_x*o1_x-w1_y*o1_y ]
        // H_S1 = [ -w1_y  w1_x     w1_x*o1_y-w1_y*o1_x ]
        //        [                         f_S1        ]

        // compute H_S2 on target plane based on TAP, 9 flops
		float o2_x = 0.5 * (tar[0] + tar[2]);   // midpoint O2, 2 flops
		float o2_y = 0.5 * (tar[1] + tar[3]);   // 2
		float w2_x = o2_x - tar[0];             // 1, vector O2N2
		float w2_y = tar[1] - o2_y;             // 1
		float f_S2 = (w2_x * w2_x + w2_y * w2_y); // 3
        //            [ w2_x -w2_y o2_x ]
        // H_S2_inv = [ w2_y  w2_x o2_y ]
        //            [              1  ]

        // compute P3 (8 flops) and P5 (6 flops)
        //
        //                              [ 1       ]   [  w1_x  w1_y       ]   [ 1     -o1_x  ]   [ P1_x ]   [ P5_x ]
        // P5 = H_E*H_S1*P1 = H_E*P3 =  [       1 ] * [ -w1_y  w1_x       ] * [    1  -o1_y  ] * [ P1_y ] = [ P5_y ]
        //                              [    1    ]   [              f_S1 ]   [          1   ]   [   1  ]   [   1  ]

		float w3_x = src[4] - o1_x;     // 1
		float w3_y = src[5] - o1_y;     // 1
		float p3_x = w1_x * w3_x - w1_y * w3_y;     // 3
		float p3_y = w1_y * w3_x + w1_x * w3_y;     // 3, p5 = [p3_x f1 p3_y]
		float temp1 = 1.0 / p3_y;       // 4
		float p5_x = temp1 * p3_x;      // 1
		float p5_y = temp1 * f_S1;      // 1

        // compute Q3 (8 flops) and Q7 (6 flops)
		float w5_x = src[6] - o1_x;     // 1
		float w5_y = src[7] - o1_y;     // 1
		float q3_x = w1_x * w5_x - w1_y * w5_y;    // 3
		float q3_y = w1_y * w5_x + w1_x * w5_y;    // 3, q5 = [q3_x f1 q3_y]
		float q7_x = p3_y * q3_x - p3_x * q3_y;    // 3
		float q7_y = (p3_y - q3_y) * f_S1;         // 2
		float q7_f = p3_y * q3_y;		// 1, q7 = [q7_x q7_y q7_f]
        //                [ p3_y   0    -p3_x ]         [ p3_y * q3_x - p3_x * q3_y ]
        // Q7 = H_T1*Q5 = [  0    p3_y  -f_S1 ] * Q5 =  [   (p3_y - q3_y) * f_S1    ]
        //                [  0     0     p3_y ]         [        p3_y * q3_y        ]

        // compute P4 (8 flops) and P6 (6 flops)
		float w4_x = tar[4] - o2_x;     // 1
		float w4_y = tar[5] - o2_y;     // 1
		float p4_x = w2_x * w4_x - w2_y * w4_y;     // 3
		float p4_y = w2_y * w4_x + w2_x * w4_y;     // 3, p6 = [p4_x f2 p4_y]
		float temp2 = 1.0 / p4_y;       // 4
		float p6_x = temp2 * p4_x;      // 1
		float p6_y = temp2 * f_S2;      // 1

        // compute Q4 (8 flops) and Q8 (6 flops)
		float w6_x = tar[6] - o2_x;     // 1
		float w6_y = tar[7] - o2_y;     // 1
		float q4_x = w2_x * w6_x - w2_y * w6_y;     // 3
		float q4_y = w2_y * w6_x + w2_x * w6_y;     // 3, q6 = [q4_x f2 q4_y]
		float q8_x = p4_y * q4_x - p4_x * q4_y;     // 3
		float q8_y = (p4_y - q4_y) * f_S2;          // 2
		float q8_f = p4_y * q4_y;                   // 1, q8 = [q8_x q8_y q8_f]

		// solving the four variables in H_K, 24 flops
        //                   [ a_K   b_K   0 ]
        //     Q8 = H_G*Q7 = [ b_K   a_K   0 ] * Q7
        //                   [  0     0    1 ]
        // consequently, we have
        //     Q7_x*a_K + Q7_y*b_K = Q8_x*Q7_f/Q8_f;
        //     Q7_x*b_K + Q7_y*a_K = Q8_x*Q7_f/Q8_x;
        // then, a_K and b_K can be obtained.
		temp1 = q7_x * q8_x - q7_y * q8_y;      // 3
		temp2 = q7_x * q8_y - q7_y * q8_x;      // 3
		float temp3 = q7_x * q7_x - q7_y * q7_y;        // 3
		temp3 = q7_f / (temp3 * q8_f);      // 5
		float a_K = temp1 * temp3;        // 1
		float b_K = temp2 * temp3;        // 1
		float u_K = p6_x - a * p5_x - b * p5_y;    // 4
		float v_K = p6_y - a * p5_y - b * p5_x;    // 4

        // compute the first two rows H_L = H_S2_inv * H_K, 0 flops
        //          [ b_K * o2_x + a_K * w2_x    w2_y + o2_x * v_K + u_K * w2_x    a_K * o2_x + b_K * w2_x ]
        // H_L =    [ b_K * o2_y - a_K * w2_y    w2_x + o2_y * v_K - u_K * w2_y    a_K * o2_y - b_K * w2_y ]
        //          [          b_K                           v_K                             a_K           ]
		float H_L[9] = { b_K * o2_x + a_K * w2_x, w2_y + o2_x * v_K + u_K * w2_x, a_K * o2_x + b_K * w2_x,
                         b_K * o2_y - a_K * w2_y, w2_x + o2_y * v_K - u_K * w2_y, a_K * o2_y - b_K * w2_y,
                         b_K, v_K, a_K };

        // compute the two elements in H_S1, 6 flops
		float S1_13 = w1_y * o1_y - w1_x * o1_x;
		float S1_23 = -w1_y * o1_x - w1_x * o1_y;

        // compute homography up to a scale, H = H_L * H_S1, 33 flops
		result[0] = H_L[0] * w1_x + H_L[1] * w1_y;
		result[1] = H_L[1] * w1_x - H_L[0] * w1_y;
		result[2] = H_L[2] * f_S1 + H_L[0] * S1_13 + H_L[1] * S1_23;
		result[3] = H_L[3] * w1_x + H_L[4] * w1_y;
		result[4] = H_L[4] * w1_x - H_L[3] * w1_y;
		result[5] = H_L[5] * f_S1 + H_L[3] * S1_13 + H_L[4] * S1_23;
		result[6] = H_L[6] * w1_x + H_L[7] * w1_y;
		result[7] = H_L[7] * w1_x - H_L[6] * w1_y;
		result[8] = H_L[8] * f_S1 + H_L[6] * S1_13 + H_L[7] * S1_23;

        // normalization based on the last element of H, 12 flops
		float result8_ = 1 / result[8];
		for (int i = 0; i < 8; i++) {
			result[i] *= result8_;
		}
		result[8] = 1.0f;

		return 0;
	}

	int runKernel_SKS_double(float* src, float* tar, double* result) {
		// compute the similarity transformation H_S1 on source plane based on two anchor points (e.g., M1 and N1), 9 flops
		double o1_x = 0.5 * (src[0] + src[2]);   // midpoint O1, 2 flops
		double o1_y = 0.5 * (src[1] + src[3]);   // 2
		double w1_x = o1_x - src[0];             // 1, vector O1N1 = M1O1
		double w1_y = src[1] - o1_y;             // 1
		double f_S1 = (w1_x * w1_x + w1_y * w1_y); // 3
        //        [  w1_x  w1_y    -w1_x*o1_x-w1_y*o1_y ]
        // H_S1 = [ -w1_y  w1_x     w1_x*o1_y-w1_y*o1_x ]
        //        [                         f_S1        ]

        // compute H_S2 on target plane based on TAP, 9 flops
		double o2_x = 0.5 * (tar[0] + tar[2]);   // midpoint O2, 2 flops
		double o2_y = 0.5 * (tar[1] + tar[3]);   // 2
		double w2_x = o2_x - tar[0];             // 1, vector O2N2
		double w2_y = tar[1] - o2_y;             // 1
		double f_S2 = (w2_x * w2_x + w2_y * w2_y); // 3
        //            [ w2_x -w2_y o2_x ]
        // H_S2_inv = [ w2_y  w2_x o2_y ]
        //            [              1  ]

        // compute P3 (8 flops) and P5 (6 flops)
        //
        //                              [ 1       ]   [  w1_x  w1_y       ]   [ 1     -o1_x  ]   [ P1_x ]   [ P5_x ]
        // P5 = H_E*H_S1*P1 = H_E*P3 =  [       1 ] * [ -w1_y  w1_x       ] * [    1  -o1_y  ] * [ P1_y ] = [ P5_y ]
        //                              [    1    ]   [              f_S1 ]   [          1   ]   [   1  ]   [   1  ]

		double w3_x = src[4] - o1_x;     // 1
		double w3_y = src[5] - o1_y;     // 1
		double p3_x = w1_x * w3_x - w1_y * w3_y;     // 3
		double p3_y = w1_y * w3_x + w1_x * w3_y;     // 3, p5 = [p3_x f1 p3_y]
		double temp1 = 1.0 / p3_y;       // 4
		double p5_x = temp1 * p3_x;      // 1
		double p5_y = temp1 * f_S1;      // 1

        // compute Q3 (8 flops) and Q7 (6 flops)
		double w5_x = src[6] - o1_x;     // 1
		double w5_y = src[7] - o1_y;     // 1
		double q3_x = w1_x * w5_x - w1_y * w5_y;    // 3
		double q3_y = w1_y * w5_x + w1_x * w5_y;    // 3, q5 = [q3_x f1 q3_y]
		double q7_x = p3_y * q3_x - p3_x * q3_y;    // 3
		double q7_y = (p3_y - q3_y) * f_S1;         // 2
		double q7_f = p3_y * q3_y;		// 1, q7 = [q7_x q7_y q7_f]
        //                [ p3_y   0    -p3_x ]         [ p3_y * q3_x - p3_x * q3_y ]
        // Q7 = H_T1*Q5 = [  0    p3_y  -f_S1 ] * Q5 =  [   (p3_y - q3_y) * f_S1    ]
        //                [  0     0     p3_y ]         [        p3_y * q3_y        ]

        // compute P4 (8 flops) and P6 (6 flops)
		double w4_x = tar[4] - o2_x;     // 1
		double w4_y = tar[5] - o2_y;     // 1
		double p4_x = w2_x * w4_x - w2_y * w4_y;     // 3
		double p4_y = w2_y * w4_x + w2_x * w4_y;     // 3, p6 = [p4_x f2 p4_y]
		double temp2 = 1.0 / p4_y;       // 4
		double p6_x = temp2 * p4_x;      // 1
		double p6_y = temp2 * f_S2;      // 1

        // compute Q4 (8 flops) and Q8 (6 flops)
		double w6_x = tar[6] - o2_x;     // 1
		double w6_y = tar[7] - o2_y;     // 1
		double q4_x = w2_x * w6_x - w2_y * w6_y;     // 3
		double q4_y = w2_y * w6_x + w2_x * w6_y;     // 3, q6 = [q4_x f2 q4_y]
		double q8_x = p4_y * q4_x - p4_x * q4_y;     // 3
		double q8_y = (p4_y - q4_y) * f_S2;          // 2
		double q8_f = p4_y * q4_y;                   // 1, q8 = [q8_x q8_y q8_f]

		// solving the four variables in H_K, 24 flops
        //                   [ a_K   b_K   0 ]
        //     Q8 = H_G*Q7 = [ b_K   a_K   0 ] * Q7
        //                   [  0     0    1 ]
        // consequently, we have
        //     Q7_x*a_K + Q7_y*b_K = Q8_x*Q7_f/Q8_f;
        //     Q7_x*b_K + Q7_y*a_K = Q8_x*Q7_f/Q8_x;
        // then, a_K and b_K can be obtained.
		temp1 = q7_x * q8_x - q7_y * q8_y;      // 3
		temp2 = q7_x * q8_y - q7_y * q8_x;      // 3
		double temp3 = q7_x * q7_x - q7_y * q7_y;        // 3
		temp3 = q7_f / (temp3 * q8_f);      // 5
		double a_K = temp1 * temp3;        // 1
		double b_K = temp2 * temp3;        // 1
		double u_K = p6_x - a * p5_x - b * p5_y;    // 4
		double v_K = p6_y - a * p5_y - b * p5_x;    // 4

        // compute the first two rows H_L = H_S2_inv * H_K, 0 flops
        //          [ b_K * o2_x + a_K * w2_x    w2_y + o2_x * v_K + u_K * w2_x    a_K * o2_x + b_K * w2_x ]
        // H_L =    [ b_K * o2_y - a_K * w2_y    w2_x + o2_y * v_K - u_K * w2_y    a_K * o2_y - b_K * w2_y ]
        //          [          b_K                           v_K                             a_K           ]
		double H_L[9] = { b_K * o2_x + a_K * w2_x, w2_y + o2_x * v_K + u_K * w2_x, a_K * o2_x + b_K * w2_x,
                         b_K * o2_y - a_K * w2_y, w2_x + o2_y * v_K - u_K * w2_y, a_K * o2_y - b_K * w2_y,
                         b_K, v_K, a_K };

        // compute the two elements in H_S1, 6 flops
		double S1_13 = w1_y * o1_y - w1_x * o1_x;
		double S1_23 = -w1_y * o1_x - w1_x * o1_y;

        // compute homography up to a scale, H = H_L * H_S1, 33 flops
		result[0] = H_L[0] * w1_x + H_L[1] * w1_y;
		result[1] = H_L[1] * w1_x - H_L[0] * w1_y;
		result[2] = H_L[2] * f_S1 + H_L[0] * S1_13 + H_L[1] * S1_23;
		result[3] = H_L[3] * w1_x + H_L[4] * w1_y;
		result[4] = H_L[4] * w1_x - H_L[3] * w1_y;
		result[5] = H_L[5] * f_S1 + H_L[3] * S1_13 + H_L[4] * S1_23;
		result[6] = H_L[6] * w1_x + H_L[7] * w1_y;
		result[7] = H_L[7] * w1_x - H_L[6] * w1_y;
		result[8] = H_L[8] * f_S1 + H_L[6] * S1_13 + H_L[7] * S1_23;

        // normalization based on the last element of H, 12 flops
		double result8_ = 1 / result[8];
		for (int i = 0; i < 8; i++) {
			result[i] *= result8_;
		}
		result[8] = 1.0f;

		return 0;
	}
}