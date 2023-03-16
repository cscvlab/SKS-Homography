#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include<opencv2/opencv.hpp>
#include<curand.h>

#define CHECK(call)\
{\
	const cudaError_t error=call;\
	if(error!=cudaSuccess)\
	{\
		printf("ERROE: %s:%d,", __FILE__,__LINE__);\
		printf("Code:%d,reason:%s\n",error,cudaGetErrorString(error));\
		exit(1);\
	}\
}

#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

using namespace cv;
using namespace std;



void read_points(std::string filename, std::vector<Point2f>& points1, std::vector<Point2f>& points2) {
	char buffer[50];
	int nump;

	Point2f tmp1, tmp2;
	std::ifstream orig_pts(filename);
	orig_pts.getline(buffer, 50);
	sscanf_s(buffer, "%d", &nump);
	for (int i = 0; i < nump; i++) {
		orig_pts.getline(buffer, 50);
		sscanf_s(buffer, "%f %f %f %f", &tmp1.x, &tmp1.y, &tmp2.x, &tmp2.y);
		points1.push_back(tmp1);
		points2.push_back(tmp2);
	}
	orig_pts.close();
}


__device__ double fabs(double x);
__device__ double sqrt(double x);

__global__ void get_rand_list(unsigned int* randList, unsigned int size, Point2d* src, Point2d* tar, double* d_src, double* d_tar, int res_size) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= res_size)
		return;
	unsigned int r1 = randList[id] % size;
	unsigned int r2 = randList[id + 1 * res_size] % size;
	unsigned int r3 = randList[id + 2 * res_size] % size;
	unsigned int r4 = randList[id + 3 * res_size] % size;

	d_src[id] = src[r1].x;
	d_src[id + 1 * res_size] = src[r1].y;
	d_src[id + 2 * res_size] = src[r2].x;
	d_src[id + 3 * res_size] = src[r2].y;
	d_src[id + 4 * res_size] = src[r3].x;
	d_src[id + 5 * res_size] = src[r3].y;
	d_src[id + 6 * res_size] = src[r4].x;
	d_src[id + 7 * res_size] = src[r4].y;

	d_tar[id] = tar[r1].x;
	d_tar[id + 1 * res_size] = tar[r1].y;
	d_tar[id + 2 * res_size] = tar[r2].x;
	d_tar[id + 3 * res_size] = tar[r2].y;
	d_tar[id + 4 * res_size] = tar[r3].x;
	d_tar[id + 5 * res_size] = tar[r3].y;
	d_tar[id + 6 * res_size] = tar[r4].x;
	d_tar[id + 7 * res_size] = tar[r4].y;
}


__global__ void cal_Homo_ACA(double* src, double* tar, double* result, int offset) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= offset)
		return;


	double SRC[8] = { src[id + 0 * offset], src[id + 1 * offset], src[id + 2 * offset],
					 src[id + 3 * offset], src[id + 4 * offset], src[id + 5 * offset],
					 src[id + 6 * offset], src[id + 7 * offset] };



	double TAR[8] = { tar[id + 0 * offset], tar[id + 1 * offset], tar[id + 2 * offset],
					 tar[id + 3 * offset], tar[id + 4 * offset], tar[id + 5 * offset],
					 tar[id + 6 * offset], tar[id + 7 * offset] };


	double M1N1_x = SRC[2] - SRC[0], M1N1_y = SRC[3] - SRC[1];
	double M1P1_x = SRC[4] - SRC[0], M1P1_y = SRC[5] - SRC[1];
	double M1Q1_x = SRC[6] - SRC[0], M1Q1_y = SRC[7] - SRC[1];
	
	double fA1 = M1N1_x * M1P1_y - M1N1_y * M1P1_x;
	double Q3_x = M1P1_y * M1Q1_x - M1P1_x * M1Q1_y;
	double Q3_y = M1N1_x * M1Q1_y - M1N1_y * M1Q1_x;


	double M2N2_x = TAR[2] - TAR[0], M2N2_y = TAR[3] - TAR[1];
	double M2P2_x = TAR[4] - TAR[0], M2P2_y = TAR[5] - TAR[1];
	double M2Q2_x = TAR[6] - TAR[0], M2Q2_y = TAR[7] - TAR[1];
	double fA2 = M2N2_x * M2P2_y - M2N2_y * M2P2_x;
	double Q4_x = M2P2_y * M2Q2_x - M2P2_x * M2Q2_y;
	double Q4_y = M2N2_x * M2Q2_y - M2N2_y * M2Q2_x;

	double tt1 = fA1 - Q3_x - Q3_y; 
	double C11 = Q3_y * Q4_x * tt1;
	double C22 = Q3_x * Q4_y * tt1;
	double C33 = Q3_x * Q3_y * (fA2 - Q4_x - Q4_y);
	double C31 = C11 - C33;
	double C32 = C22 - C33;


	double tt3 = TAR[0] * C33;
	double tt4 = TAR[1] * C33;
	double H1_11 = TAR[2] * C11 - tt3;
	double H1_12 = TAR[4] * C22 - tt3;
	double H1_21 = TAR[3] * C11 - tt4;
	double H1_22 = TAR[5] * C22 - tt4;

	double H[9];
	H[0] = H1_11 * M1P1_y - H1_12 * M1N1_y;
	H[1] = H1_12 * M1N1_x - H1_11 * M1P1_x;
	H[2] = tt3 * fA1 - H[0] * SRC[0] - H[1] * SRC[1];
	H[3] = H1_21 * M1P1_y - H1_22 * M1N1_y;
	H[4] = H1_22 * M1N1_x - H1_21 * M1P1_x;
	H[5] = tt4 * fA1 - H[3] * SRC[0] - H[4] * SRC[1];
	H[6] = C31 * M1P1_y - C32 * M1N1_y;
	H[7] = C32 * M1N1_x - C31 * M1P1_x;
	H[8] = C33 * fA1 - H[6] * SRC[0] - H[7] * SRC[1];


	result[id] = H[0];
	result[id + 1 * offset] = H[1];
	result[id + 2 * offset] = H[2];
	result[id + 3 * offset] = H[3];
	result[id + 4 * offset] = H[4];
	result[id + 5 * offset] = H[5];
	result[id + 6 * offset] = H[6];
	result[id + 7 * offset] = H[7];
	result[id + 8 * offset] = H[8];

}

__global__ void cal_Homo_SKS(double* src, double* tar, double* result, int offset) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= offset)
		return;

	double SRC[8] = { src[id + 0 * offset], src[id + 1 * offset], src[id + 2 * offset],
					 src[id + 3 * offset], src[id + 4 * offset], src[id + 5 * offset],
					 src[id + 6 * offset], src[id + 7 * offset] };


	double TAR[8] = { tar[id + 0 * offset], tar[id + 1 * offset], tar[id + 2 * offset],
					 tar[id + 3 * offset], tar[id + 4 * offset], tar[id + 5 * offset],
					 tar[id + 6 * offset], tar[id + 7 * offset] };


	double o1_x = 0.5 * (SRC[0] + SRC[2]);
	double o1_y = 0.5 * (SRC[1] + SRC[3]);
	double w1_x = o1_x - SRC[0];
	double w1_y = SRC[1] - o1_y;
	double f_S1 = (w1_x * w1_x + w1_y * w1_y);

	double o2_x = 0.5 * (TAR[0] + TAR[2]);
	double o2_y = 0.5 * (TAR[1] + TAR[3]);
	double w2_x = o2_x - TAR[0];
	double w2_y = TAR[1] - o2_y;
	double f_S2 = (w2_x * w2_x + w2_y * w2_y);

	double w3_x = SRC[4] - o1_x;
	double w3_y = SRC[5] - o1_y;
	double p3_x = w1_x * w3_x - w1_y * w3_y;
	double p3_y = w1_y * w3_x + w1_x * w3_y;
	double temp1 = 1.0 / p3_y;
	double p5_x = temp1 * p3_x;
	double p5_y = temp1 * f_S1;

	double w5_x = SRC[6] - o1_x;
	double w5_y = SRC[7] - o1_y;
	double q3_x = w1_x * w5_x - w1_y * w5_y;
	double q3_y = w1_y * w5_x + w1_x * w5_y;

	double q7_x = p3_y * q3_x - p3_x * q3_y;
	double q7_y = (p3_y - q3_y) * f_S1;
	double q7_f = p3_y * q3_y;

	double w4_x = TAR[4] - o2_x;
	double w4_y = TAR[5] - o2_y;
	double p4_x = w2_x * w4_x - w2_y * w4_y;
	double p4_y = w2_y * w4_x + w2_x * w4_y;
	double temp2 = 1.0 / p4_y;
	double p6_x = temp2 * p4_x;
	double p6_y = temp2 * f_S2;

	double w6_x = TAR[6] - o2_x;
	double w6_y = TAR[7] - o2_y;
	double q4_x = w2_x * w6_x - w2_y * w6_y;
	double q4_y = w2_y * w6_x + w2_x * w6_y;

	double q8_x = p4_y * q4_x - p4_x * q4_y;
	double q8_y = (p4_y - q4_y) * f_S2;
	double q8_f = p4_y * q4_y;

	temp1 = q7_x * q8_x - q7_y * q8_y;
	temp2 = q7_x * q8_y - q7_y * q8_x;
	double temp3 = q7_x * q7_x - q7_y * q7_y;
	temp3 = q7_f / (temp3 * q8_f);
	double a_K = temp1 * temp3;
	double b_K = temp2 * temp3;
	double u_K = p6_x - a_K * p5_x - b_K * p5_y;
	double v_K = p6_y - a_K * p5_y - b_K * p5_x;

	double H_L[9] = { b_K * o2_x + a_K * w2_x, w2_y + o2_x * v_K + u_K * w2_x, a_K * o2_x + b_K * w2_x, 
						b_K * o2_y - a_K * w2_y, w2_x + o2_y * v_K - u_K * w2_y, a_K * o2_y - b_K * w2_y, 
						b_K, v_K, a_K };


	double S1_13 = w1_y * o1_y - w1_x * o1_x;
	double S1_23 = -w1_y * o1_x - w1_x * o1_y;
	result[id] = H_L[0] * w1_x + H_L[1] * w1_y;
	result[id + 1 * offset] = H_L[1] * w1_x - H_L[0] * w1_y;
	result[id + 2 * offset] = H_L[2] * f_S1 + H_L[0] * S1_13 + H_L[1] * S1_23;
	result[id + 3 * offset] = H_L[3] * w1_x + H_L[4] * w1_y;
	result[id + 4 * offset] = H_L[4] * w1_x - H_L[3] * w1_y;
	result[id + 5 * offset] = H_L[5] * f_S1 + H_L[3] * S1_13 + H_L[4] * S1_23;
	result[id + 6 * offset] = H_L[6] * w1_x + H_L[7] * w1_y;
	result[id + 7 * offset] = H_L[7] * w1_x - H_L[6] * w1_y;
	result[id + 8 * offset] = H_L[8] * f_S1 + H_L[6] * S1_13 + H_L[7] * S1_23;

}

__device__ void scaleIndex(double* matrix, int n, int index) {
	int start = index * n + index;
	int end = index * n + n;
	for (int i = start + 1; i < end; i++) {
		matrix[i] = (matrix[i] / matrix[start]);
	}
}
__device__ void eliminate(double* A, int n, int index, int bsize) {
	double pivot[8] = { 0 };
	for (int i = index; i < n; i++)
		pivot[i] = A[(index * n) + i];
	for (int i = index + 1; i < n; i++) {
		int currentRow = i * n;
		int start = currentRow + index;
		int end = currentRow + n;
		for (int j = start + 1; j < end; j++)
			A[j] = A[j] - (A[start] * pivot[j - currentRow]);
	}
}

__device__ void find_pivot(double* A, double* B, int row_id, int numsofRow, int numsofCol) {
	double maxPivot = fabs(A[numsofCol * row_id + row_id]);
	int maxIndex = row_id;
	for (int i = row_id + 1; i < numsofRow; i++)
		if (maxPivot < fabs(A[numsofCol * i + row_id])) {
			maxIndex = i;
			maxPivot = fabs(A[numsofCol * i + row_id]);
		}

	double tmp;
	for (int i = 0; i < numsofCol; i++) {
		tmp = A[numsofCol * row_id + i];
		A[numsofCol * row_id + i] = A[numsofCol * maxIndex + i];
		A[numsofCol * maxIndex + i] = tmp;
	}
	tmp = B[row_id];
	B[row_id] = B[maxIndex];
	B[maxIndex] = tmp;
}

__device__ void down_tri_solve(double* A, double* B) {
	B[0] = B[0] / A[0 * 8 + 0];
	B[1] = (B[1] - A[1 * 8 + 0] * B[0]) / A[1 * 8 + 1];
	B[2] = (B[2] - A[2 * 8 + 0] * B[0] - A[2 * 8 + 1] * B[1]) / A[2 * 8 + 2];
	B[3] = (B[3] - A[3 * 8 + 0] * B[0] - A[3 * 8 + 1] * B[1] - A[3 * 8 + 2] * B[2]) / A[3 * 8 + 3];
	B[4] = (B[4] - A[4 * 8 + 0] * B[0] - A[4 * 8 + 1] * B[1] - A[4 * 8 + 2] * B[2] - A[4 * 8 + 3] * B[3]) / A[4 * 8 + 4];
	B[5] = (B[5] - A[5 * 8 + 0] * B[0] - A[5 * 8 + 1] * B[1] - A[5 * 8 + 2] * B[2] - A[5 * 8 + 3] * B[3] - A[5 * 8 + 4] * B[4]) / A[5 * 8 + 5];
	B[6] = (B[6] - A[6 * 8 + 0] * B[0] - A[6 * 8 + 1] * B[1] - A[6 * 8 + 2] * B[2] - A[6 * 8 + 3] * B[3] - A[6 * 8 + 4] * B[4] - A[6 * 8 + 5] * B[5]) / A[6 * 8 + 6];
	B[7] = (B[7] - A[7 * 8 + 0] * B[0] - A[7 * 8 + 1] * B[1] - A[7 * 8 + 2] * B[2] - A[7 * 8 + 3] * B[3] - A[7 * 8 + 4] * B[4] - A[7 * 8 + 5] * B[5] - A[7 * 8 + 6] * B[6]) / A[7 * 8 + 7];
}
__device__ void up_tri_solve(double* A, double* B) {
	B[6] = (B[6] - A[6 * 8 + 7] * B[7]);
	B[5] = (B[5] - A[5 * 8 + 7] * B[7] - A[5 * 8 + 6] * B[6]);
	B[4] = (B[4] - A[4 * 8 + 7] * B[7] - A[4 * 8 + 6] * B[6] - A[4 * 8 + 5] * B[5]);
	B[3] = (B[3] - A[3 * 8 + 7] * B[7] - A[3 * 8 + 6] * B[6] - A[3 * 8 + 5] * B[5] - A[3 * 8 + 4] * B[4]);
	B[2] = (B[2] - A[2 * 8 + 7] * B[7] - A[2 * 8 + 6] * B[6] - A[2 * 8 + 5] * B[5] - A[2 * 8 + 4] * B[4] - A[2 * 8 + 3] * B[3]);
	B[1] = (B[1] - A[1 * 8 + 7] * B[7] - A[1 * 8 + 6] * B[6] - A[1 * 8 + 5] * B[5] - A[1 * 8 + 4] * B[4] - A[1 * 8 + 3] * B[3] - A[1 * 8 + 2] * B[2]);
	B[0] = (B[0] - A[0 * 8 + 7] * B[7] - A[0 * 8 + 6] * B[6] - A[0 * 8 + 5] * B[5] - A[0 * 8 + 4] * B[4] - A[0 * 8 + 3] * B[3] - A[0 * 8 + 2] * B[2] - A[0 * 8 + 1] * B[1]);
}
__global__ void cal_Homo_GPT(double* src, double* tar, double* result, int offset) {
	size_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= offset)
		return;

	double SRC[8] = { src[id + 0 * offset], src[id + 1 * offset], src[id + 2 * offset], src[id + 3 * offset],
					  src[id + 4 * offset], src[id + 5 * offset], src[id + 6 * offset], src[id + 7 * offset] };


	double TAR[8] = { tar[id + 0 * offset], tar[id + 1 * offset], tar[id + 2 * offset], tar[id + 3 * offset],
					  tar[id + 4 * offset], tar[id + 5 * offset], tar[id + 6 * offset], tar[id + 7 * offset] };

	double a[64] = { 0 }, b[8] = { 0 };


	for (int i = 0; i < 4; i++) {
		a[i * 8 + 0] = a[(i + 4) * 8 + 3] = SRC[i * 2];
		a[i * 8 + 1] = a[(i + 4) * 8 + 4] = SRC[i * 2 + 1];
		a[i * 8 + 2] = a[(i + 4) * 8 + 5] = 1;

		a[i * 8 + 3] = a[i * 8 + 4] = a[i * 8 + 5] = a[(i + 4) * 8 + 0] = a[(i + 4) * 8 + 1] = a[(i + 4) * 8 + 2] = 0;

		a[i * 8 + 6] = -SRC[i * 2] * TAR[i * 2];
		a[i * 8 + 7] = -SRC[i * 2 + 1] * TAR[i * 2];
		a[(i + 4) * 8 + 6] = -SRC[i * 2] * TAR[i * 2 + 1];
		a[(i + 4) * 8 + 7] = -SRC[i * 2 + 1] * TAR[i * 2 + 1];
		b[i] = TAR[i * 2];
		b[i + 4] = TAR[i * 2 + 1];
	}

	int bsize = 8;

	//LU decompose
	for (int i = 0; i < 8; i++) {
		find_pivot(a, b, i, 8, 8);
		scaleIndex(a, 8, i);
		eliminate(a, 8, i, bsize);
	}


	// Ly = b;
	down_tri_solve(a, b);
	// Ux = y
	up_tri_solve(a, b);


	result[id] = b[0];
	result[id + 1 * offset] = b[1];
	result[id + 2 * offset] = b[2];
	result[id + 3 * offset] = b[3];
	result[id + 4 * offset] = b[4];
	result[id + 5 * offset] = b[5];
	result[id + 6 * offset] = b[6];
	result[id + 7 * offset] = b[7];
	result[id + 8 * offset] = 1.0;

}

__global__ void cal_Homo_GE(double* src, double* tar, double* result, int offset) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= offset)
		return;

	double x0, y0, x1, y1, x2, y2, x3, y3, X0, Y0, X1, Y1, X2, Y2, X3, Y3;
	double x0X0, x2X2, x0Y0, x2Y2, y0X0, y2X2, y0Y0, y2Y2, x1X1, x3X3, x1Y1, x3Y3, y1X1, y3X3, y1Y1, y3Y3;
	double scalar1, scalar2;

	x0 = src[id];
	y0 = src[id + 1 * offset];
	x1 = src[id + 2 * offset];
	y1 = src[id + 3 * offset];
	x2 = src[id + 4 * offset];
	y2 = src[id + 5 * offset];
	x3 = src[id + 6 * offset];
	y3 = src[id + 7 * offset];
	X0 = tar[id];
	Y0 = tar[id + 1 * offset];
	X1 = tar[id + 2 * offset];
	Y1 = tar[id + 3 * offset];
	X2 = tar[id + 4 * offset];
	Y2 = tar[id + 5 * offset];
	X3 = tar[id + 6 * offset];
	Y3 = tar[id + 7 * offset];


	x0X0 = x0 * X0, x1X1 = x1 * X1, x2X2 = x2 * X2, x3X3 = x3 * X3;
	x0Y0 = x0 * Y0, x1Y1 = x1 * Y1, x2Y2 = x2 * Y2, x3Y3 = x3 * Y3;
	y0X0 = y0 * X0, y1X1 = y1 * X1, y2X2 = y2 * X2, y3X3 = y3 * X3;
	y0Y0 = y0 * Y0, y1Y1 = y1 * Y1, y2Y2 = y2 * Y2, y3Y3 = y3 * Y3;

	double minor[2][4] = { { x0 - x2, x1 - x2, x2, x3 - x2 },
							{ y0 - y2, y1 - y2, y2, y3 - y2 } };


	double major[3][8] = { { x2X2 - x0X0, x2X2 - x1X1, -x2X2, x2X2 - x3X3, x2Y2 - x0Y0, x2Y2 - x1Y1, -x2Y2, x2Y2 - x3Y3 },
							{ y2X2 - y0X0, y2X2 - y1X1, -y2X2, y2X2 - y3X3, y2Y2 - y0Y0, y2Y2 - y1Y1, -y2Y2, y2Y2 - y3Y3 },
							{ (X0 - X2), (X1 - X2), (X2), (X3 - X2), (Y0 - Y2), (Y1 - Y2), (Y2), (Y3 - Y2) } };


	scalar1 = minor[0][0], scalar2 = minor[0][1];
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




	result[id] = major[2][0];
	result[id + 1 * offset] = major[2][1];
	result[id + 2 * offset] = major[2][2];
	result[id + 3 * offset] = major[2][4];
	result[id + 4 * offset] = major[2][5];
	result[id + 5 * offset] = major[2][6];
	result[id + 6 * offset] = major[2][7];
	result[id + 7 * offset] = major[2][3];
	result[id + 8 * offset] = 1.0;
}


__host__ __device__ static double PYTHAG(double a, double b)
{
	double at = fabs(a), bt = fabs(b), ct, result;
	if (at > bt) {
		ct = bt / at;
		result = at * sqrt(1.0 + ct * ct);
	}
	else if (bt > 0.0) {
		ct = at / bt;
		result = bt * sqrt(1.0 + ct * ct);
	}
	else
		result = 0.0;

	return(result);
}

__host__ __device__ int svd(double* A, double* diag, double* v, double* rv1, int m, int n) {

	int i, j, k, l;
	double f, h, s;
	double anorm = 0.0, g = 0.0, scale = 0.0;

	for (i = 0; i < n; i++)
	{
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m)
		{
			for (k = i; k < m; k++)//array[i][j]=*(array +i*n +j)
								   //scale += fabs(A[k][i]);//A[k][i]=*(A+k*n+i)
				scale += fabs(*(A + k * n + i));
			if (scale)
			{
				for (k = i; k < m; k++)
				{
					*(A + k * n + i) = (*(A + k * n + i) / scale);
					s += (*(A + k * n + i) * *(A + k * n + i));
				}
				f = *(A + i * n + i);
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				*(A + i * n + i) = f - g;
				if (i != n - 1)
				{
					for (j = l; j < n; j++)
					{
						for (s = 0.0, k = i; k < m; k++)
							s += *(A + k * n + i) * *(A + k * n + j);
						f = s / h;
						for (k = i; k < m; k++)
							*(A + k * n + j) += f * *(A + k * n + i);
					}
				}
				for (k = i; k < m; k++)
					*(A + k * n + i) = *(A + k * n + i) * scale;
			}
		}
		diag[i] = scale * g;

		/* right-hand reduction */
		g = s = scale = 0.0;
		if (i < m && i != n - 1)
		{
			for (k = l; k < n; k++)
				scale += fabs(*(A + i * n + k));
			if (scale)
			{
				for (k = l; k < n; k++)
				{
					*(A + i * n + k) = *(A + i * n + k) / scale;
					s += *(A + i * n + k) * *(A + i * n + k);
				}
				f = *(A + i * n + l);
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				*(A + i * n + l) = f - g;
				for (k = l; k < n; k++)
					rv1[k] = *(A + i * n + k) / h;
				if (i != m - 1)
				{
					for (j = l; j < m; j++)
					{
						for (s = 0.0, k = l; k < n; k++)
							s += *(A + j * n + k) * *(A + i * n + k);
						for (k = l; k < n; k++)
							*(A + j * n + k) += s * rv1[k];
					}
				}
				for (k = l; k < n; k++)
					*(A + i * n + k) = *(A + i * n + k) * scale;
			}
		}
		anorm = MAX(anorm, (fabs(diag[i]) + fabs(rv1[i])));
	}

	/* accumulate the right-hand transformation */
	for (i = n - 1; i >= 0; i--)
	{
		if (i < n - 1)
		{
			if (g)
			{
				for (j = l; j < n; j++)
					*(v + j * n + i) = (*(A + i * n + j) / *(A + i * n + l)) / g;
				/* double division to avoid underflow */
				for (j = l; j < n; j++)
				{
					for (s = 0.0, k = l; k < n; k++)
						s += (*(A + i * n + k) * *(v + k * n + j));
					for (k = l; k < n; k++)
						*(v + k * n + j) += s * *(v + k * n + i);
				}
			}
			for (j = l; j < n; j++)
				*(v + i * n + j) = *(v + j * n + i) = 0.0;
		}
		*(v + i * n + i) = 1.0;
		g = rv1[i];
		l = i;
	}

	/* accumulate the left-hand transformation */
	for (i = n - 1; i >= 0; i--)
	{
		l = i + 1;
		g = diag[i];
		if (i < n - 1)
			for (j = l; j < n; j++)
				*(A + i * n + j) = 0.0;
		if (g)
		{
			g = 1.0 / g;
			if (i != n - 1)
			{
				for (j = l; j < n; j++)
				{
					for (s = 0.0, k = l; k < m; k++)
						s += *(A + k * n + i) * *(A + k * n + j);
					f = s / *(A + i * n + i) * g;
					for (k = i; k < m; k++)
						*(A + k * n + j) += (f * *(A + k * n + i));
				}
			}
			for (j = i; j < m; j++)
				*(A + j * n + i) = *(A + j * n + i) * g;
		}
		else
		{
			for (j = i; j < m; j++)
				*(A + j * n + i) = 0.0;
		}
		++(*(A + i * n + i));
	}
	int flag, its, jj, nm;
	double c, x, y, z;
	/* diagonalize the bidiagonal form */
	for (k = n - 1; k >= 0; k--)
	{                             /* loop over singular values */
		for (its = 0; its < 30; its++)
		{                         /* loop over allowed iterations */
			flag = 1;
			for (l = k; l >= 0; l--)
			{                     /* test for splitting */
				nm = l - 1;
				if (fabs(rv1[l]) + anorm == anorm)
				{
					flag = 0;
					break;
				}
				if (fabs(diag[nm]) + anorm == anorm)
					break;
			}
			if (flag)
			{
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++)
				{
					f = s * rv1[i];
					if (fabs(f) + anorm != anorm)
					{
						g = diag[i];
						h = PYTHAG(f, g);
						diag[i] = h;
						h = 1.0 / h;
						c = g * h;
						s = (-f * h);
						for (j = 0; j < m; j++)
						{
							y = *(A + j * n + nm);
							z = *(A + j * n + i);
							*(A + j * n + nm) = y * c + z * s;
							*(A + j * n + i) = z * c - y * s;
						}
					}
				}
			}
			z = diag[k];
			if (l == k)
			{                  /* convergence */
				if (z < 0.0)
				{              /* make singular value nonnegative */
					diag[k] = -z;
					for (j = 0; j < n; j++)
						*(v + j * n + k) = (-*(v + j * n + k));
				}
				break;
			}
			if (its >= 30) {
				free((void*)rv1);
				printf("No convergence after 30,000! iterations \n");
				return(0);
			}

			/* shift from bottom 2 x 2 minor */
			x = diag[l];
			nm = k - 1;
			y = diag[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = PYTHAG(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

			/* next QR transformation */
			c = s = 1.0;
			for (j = l; j <= nm; j++)
			{
				i = j + 1;
				g = rv1[i];
				y = diag[i];
				h = s * g;
				g = c * g;
				z = PYTHAG(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y = y * c;
				for (jj = 0; jj < n; jj++)
				{
					x = *(v + jj * n + j);
					z = *(v + jj * n + i);
					*(v + jj * n + j) = x * c + z * s;
					*(v + jj * n + i) = z * c - x * s;
				}
				z = PYTHAG(f, h);
				diag[j] = z;
				if (z)
				{
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = (c * y) - (s * g);
				for (jj = 0; jj < m; jj++)
				{
					y = *(A + jj * n + j);
					z = *(A + jj * n + i);
					*(A + jj * n + j) = y * c + z * s;
					*(A + jj * n + i) = z * c - y * s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			diag[k] = x;
		}
	}
	return(1);
}


__device__ void normalizeDataIsotropic(double* data, double* dataN, double* T, double* Ti) {

	double xm = 0, ym = 0;
	xm += data[0] + data[2] + data[4] + data[6];
	ym += data[1] + data[3] + data[5] + data[7];

	xm /= 4.0;
	ym /= 4.0;

	double kappa = 0;
	double xh, yh;

	for (int i = 0; i < 4; i++) {
		xh = data[i * 2] - xm;
		yh = data[i * 2 + 1] - ym;
		dataN[i] = xh;
		dataN[i + 4] = yh;
		kappa = kappa + xh * xh + yh * yh;
	}

	double beta = sqrt(2 * 4 / kappa);

	for (int i = 0; i < 8; i++)
		dataN[i] *= beta;

	T[0] = 1. / beta;
	T[4] = 1. / beta;
	T[2] = xm;
	T[5] = ym;
	T[8] = 1;

	Ti[0] = beta;
	Ti[4] = beta;
	Ti[2] = -beta * xm;
	Ti[5] = -beta * ym;
	Ti[8] = 1;
}
__global__ void cal_Homo_HO(double* src, double* tar, double* result, int offset) {

	size_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= offset)
		return;

	double SRC[8] = { src[id + 0 * offset], src[id + 1 * offset], src[id + 2 * offset],
					 src[id + 3 * offset], src[id + 4 * offset], src[id + 5 * offset],
					 src[id + 6 * offset], src[id + 7 * offset] };


	double TAR[8] = { tar[id + 0 * offset], tar[id + 1 * offset], tar[id + 2 * offset],
					 tar[id + 3 * offset], tar[id + 4 * offset], tar[id + 5 * offset],
					 tar[id + 6 * offset], tar[id + 7 * offset] };

	// 2*n, 2*n, 3*3, 3*3, dataAT n*2, dataADataAT 2*2;
	double dataA[8] = { 0 }, dataB[8] = { 0 }, TA[9] = { 0 }, TAi[9] = { 0 }, TB[9] = { 0 }, TBi[9] = { 0 }, dataAT[8] = { 0 }, dataADataAT[4] = { 0 };
	normalizeDataIsotropic(SRC, dataA, TA, TAi);
	normalizeDataIsotropic(TAR, dataB, TB, TBi);

	double C1[4], C2[4], C3[4], C4[4];
	double mC1 = 0, mC2 = 0, mC3 = 0, mC4 = 0;


	for (int i = 0; i < 4; i++) {
		C1[i] = -dataB[i] * dataA[i];
		C2[i] = -dataB[i] * dataA[i + 4];
		C3[i] = -dataB[i + 4] * dataA[i];
		C4[i] = -dataB[i + 4] * dataA[i + 4];
		mC1 += C1[i];
		mC2 += C2[i];
		mC3 += C3[i];
		mC4 += C4[i];
	}
	mC1 /= 4;
	mC2 /= 4;
	mC3 /= 4;
	mC4 /= 4;

	// n*3
	double Mx[12], My[12];
	for (int i = 0; i < 4; i++) {
		Mx[i * 3 + 0] = C1[i] - mC1;
		Mx[i * 3 + 1] = C2[i] - mC2;
		Mx[i * 3 + 2] = -dataB[i];

		My[i * 3 + 0] = C3[i] - mC3;
		My[i * 3 + 1] = C4[i] - mC4;
		My[i * 3 + 2] = -dataB[i + 4];
	}

	dataAT[0] = dataA[0];
	dataAT[1] = dataA[4];
	dataAT[2] = dataA[1];
	dataAT[3] = dataA[5];
	dataAT[4] = dataA[2];
	dataAT[5] = dataA[6];
	dataAT[6] = dataA[3];
	dataAT[7] = dataA[7];

	// A*A^T
	dataADataAT[0] = dataA[0] * dataAT[0] + dataA[1] * dataAT[2] + dataA[2] * dataAT[4] + dataA[3] * dataAT[6];
	dataADataAT[1] = dataA[0] * dataAT[1] + dataA[1] * dataAT[3] + dataA[2] * dataAT[5] + dataA[3] * dataAT[7];
	dataADataAT[2] = dataA[4] * dataAT[0] + dataA[5] * dataAT[2] + dataA[6] * dataAT[4] + dataA[7] * dataAT[6];
	dataADataAT[3] = dataA[4] * dataAT[1] + dataA[5] * dataAT[3] + dataA[6] * dataAT[5] + dataA[7] * dataAT[7];

	double dt = dataADataAT[0] * dataADataAT[3] - dataADataAT[1] * dataADataAT[2];
	double DataADataATi[4] = { 0 };
	DataADataATi[0] = dataADataAT[3] / dt;
	DataADataATi[1] = -dataADataAT[1] / dt;
	DataADataATi[2] = -dataADataAT[2] / dt;
	DataADataATi[3] = dataADataAT[0] / dt;

	// 2*n
	double	Pp[8] = { 0 };
	Pp[0] = DataADataATi[0] * dataA[0] + DataADataATi[1] * dataA[4];
	Pp[1] = DataADataATi[0] * dataA[1] + DataADataATi[1] * dataA[5];
	Pp[2] = DataADataATi[0] * dataA[2] + DataADataATi[1] * dataA[6];
	Pp[3] = DataADataATi[0] * dataA[3] + DataADataATi[1] * dataA[7];
	Pp[4] = DataADataATi[2] * dataA[0] + DataADataATi[3] * dataA[4];
	Pp[5] = DataADataATi[2] * dataA[1] + DataADataATi[3] * dataA[5];
	Pp[6] = DataADataATi[2] * dataA[2] + DataADataATi[3] * dataA[6];
	Pp[7] = DataADataATi[2] * dataA[3] + DataADataATi[3] * dataA[7];

	// 2*3
	double Bx[6] = { 0 }, By[6] = { 0 };
	Bx[0] = Pp[0] * Mx[0] + Pp[1] * Mx[3] + Pp[2] * Mx[6] + Pp[3] * Mx[9];
	Bx[1] = Pp[0] * Mx[1] + Pp[1] * Mx[4] + Pp[2] * Mx[7] + Pp[3] * Mx[10];
	Bx[2] = Pp[0] * Mx[2] + Pp[1] * Mx[5] + Pp[2] * Mx[8] + Pp[3] * Mx[11];
	Bx[3] = Pp[4] * Mx[0] + Pp[5] * Mx[3] + Pp[6] * Mx[6] + Pp[7] * Mx[9];
	Bx[4] = Pp[4] * Mx[1] + Pp[5] * Mx[4] + Pp[6] * Mx[7] + Pp[7] * Mx[10];
	Bx[5] = Pp[4] * Mx[2] + Pp[5] * Mx[5] + Pp[6] * Mx[8] + Pp[7] * Mx[11];

	By[0] = Pp[0] * My[0] + Pp[1] * My[3] + Pp[2] * My[6] + Pp[3] * My[9];
	By[1] = Pp[0] * My[1] + Pp[1] * My[4] + Pp[2] * My[7] + Pp[3] * My[10];
	By[2] = Pp[0] * My[2] + Pp[1] * My[5] + Pp[2] * My[8] + Pp[3] * My[11];
	By[3] = Pp[4] * My[0] + Pp[5] * My[3] + Pp[6] * My[6] + Pp[7] * My[9];
	By[4] = Pp[4] * My[1] + Pp[5] * My[4] + Pp[6] * My[7] + Pp[7] * My[10];
	By[5] = Pp[4] * Mx[2] + Pp[5] * My[5] + Pp[6] * My[8] + Pp[7] * My[11];

	// n*3
	double Ex[12] = { 0 }, Ey[12] = { 0 };
	Ex[0] = dataAT[0] * Bx[0] + dataAT[1] * Bx[3];
	Ex[1] = dataAT[0] * Bx[1] + dataAT[1] * Bx[4];
	Ex[2] = dataAT[0] * Bx[2] + dataAT[1] * Bx[5];
	Ex[3] = dataAT[2] * Bx[0] + dataAT[3] * Bx[3];
	Ex[4] = dataAT[2] * Bx[1] + dataAT[3] * Bx[4];
	Ex[5] = dataAT[2] * Bx[2] + dataAT[3] * Bx[5];
	Ex[6] = dataAT[4] * Bx[0] + dataAT[5] * Bx[3];
	Ex[7] = dataAT[4] * Bx[1] + dataAT[5] * Bx[4];
	Ex[8] = dataAT[4] * Bx[2] + dataAT[5] * Bx[5];
	Ex[9] = dataAT[6] * Bx[0] + dataAT[7] * Bx[3];
	Ex[10] = dataAT[6] * Bx[1] + dataAT[7] * Bx[4];
	Ex[11] = dataAT[6] * Bx[2] + dataAT[7] * Bx[5];

	Ey[0] = dataAT[0] * By[0] + dataAT[1] * By[3];
	Ey[1] = dataAT[0] * By[1] + dataAT[1] * By[4];
	Ey[2] = dataAT[0] * By[2] + dataAT[1] * By[5];
	Ey[3] = dataAT[2] * By[0] + dataAT[3] * By[3];
	Ey[4] = dataAT[2] * By[1] + dataAT[3] * By[4];
	Ey[5] = dataAT[2] * By[2] + dataAT[3] * By[5];
	Ey[6] = dataAT[4] * By[0] + dataAT[5] * By[3];
	Ey[7] = dataAT[4] * By[1] + dataAT[5] * By[4];
	Ey[8] = dataAT[4] * By[2] + dataAT[5] * By[5];
	Ey[9] = dataAT[6] * By[0] + dataAT[7] * By[3];
	Ey[10] = dataAT[6] * By[1] + dataAT[7] * By[4];
	Ey[11] = dataAT[6] * By[2] + dataAT[7] * By[5];

	// (2*n)*3
	double D[24] = { 0 };
	for (int i = 0; i < 4; i++) {
		D[i * 3 + 0] = Mx[i * 3 + 0] - Ex[i * 3 + 0];
		D[i * 3 + 1] = Mx[i * 3 + 1] - Ex[i * 3 + 1];
		D[i * 3 + 2] = Mx[i * 3 + 2] - Ex[i * 3 + 2];

		D[i + 12 + 0] = My[i * 3 + 0] - Ey[i * 3 + 0];
		D[i + 12 + 1] = My[i * 3 + 1] - Ey[i * 3 + 1];
		D[i + 12 + 2] = My[i * 3 + 2] - Ey[i * 3 + 2];
	}


	double diag[3], v[9], rvl[9];
	int _ = svd(D, diag, v, rvl, 8, 3);
	int index_diag = 0;
	if (diag[1] < diag[index_diag])
		index_diag = 1;
	if (diag[2] < diag[index_diag])
		index_diag = 2;

	double h789[3] = { v[index_diag], v[index_diag + 3], v[index_diag + 6] };
	double h12[2], h45[2];
	h12[0] = -1 * (Bx[0] * h789[0] + Bx[1] * h789[1] + Bx[2] * h789[2]);
	h12[1] = -1 * (Bx[3] * h789[0] + Bx[4] * h789[1] + Bx[5] * h789[2]);

	h45[0] = -1 * (By[0] * h789[0] + By[1] * h789[1] + By[2] * h789[2]);
	h45[1] = -1 * (By[3] * h789[0] + By[4] * h789[1] + By[5] * h789[2]);
	double h3 = -(mC1 * h789[0] + mC2 * h789[1]);
	double h6 = -(mC3 * h789[0] + mC4 * h789[1]);

	double res[9] = { 0 };
	res[0] = h12[0];
	res[1] = h12[1];
	res[2] = h3;
	res[3] = h45[0];
	res[4] = h45[1];
	res[5] = h6;
	res[6] = h789[0];
	res[7] = h789[1];
	res[8] = h789[2];

	double _Htemp[9] = { 0 };
	_Htemp[0] = TB[0] * res[0] + TB[1] * res[3] + TB[2] * res[6];
	_Htemp[1] = TB[0] * res[1] + TB[1] * res[4] + TB[2] * res[7];
	_Htemp[2] = TB[0] * res[2] + TB[1] * res[5] + TB[2] * res[8];
	_Htemp[3] = TB[3] * res[0] + TB[4] * res[3] + TB[5] * res[6];
	_Htemp[4] = TB[3] * res[1] + TB[4] * res[4] + TB[5] * res[7];
	_Htemp[5] = TB[3] * res[2] + TB[4] * res[5] + TB[5] * res[8];
	_Htemp[6] = TB[6] * res[0] + TB[7] * res[3] + TB[8] * res[6];
	_Htemp[7] = TB[6] * res[1] + TB[7] * res[4] + TB[8] * res[7];
	_Htemp[8] = TB[6] * res[2] + TB[7] * res[5] + TB[8] * res[8];

	double H[9] = { 0 };
	H[0] = _Htemp[0] * TAi[0] + _Htemp[1] * TAi[3] + _Htemp[2] * TAi[6];
	H[1] = _Htemp[0] * TAi[1] + _Htemp[1] * TAi[4] + _Htemp[2] * TAi[7];
	H[2] = _Htemp[0] * TAi[2] + _Htemp[1] * TAi[5] + _Htemp[2] * TAi[8];
	H[3] = _Htemp[3] * TAi[0] + _Htemp[4] * TAi[3] + _Htemp[5] * TAi[6];
	H[4] = _Htemp[3] * TAi[1] + _Htemp[4] * TAi[4] + _Htemp[5] * TAi[7];
	H[5] = _Htemp[3] * TAi[2] + _Htemp[4] * TAi[5] + _Htemp[5] * TAi[8];
	H[6] = _Htemp[6] * TAi[0] + _Htemp[7] * TAi[3] + _Htemp[8] * TAi[6];
	H[7] = _Htemp[6] * TAi[1] + _Htemp[7] * TAi[4] + _Htemp[8] * TAi[7];
	H[8] = _Htemp[6] * TAi[2] + _Htemp[7] * TAi[5] + _Htemp[8] * TAi[8];

	for (int i = 0; i < 9; i++)
		H[i] /= H[8];

	result[id] = H[0];
	result[id + 1 * offset] = H[1];
	result[id + 2 * offset] = H[2];
	result[id + 3 * offset] = H[3];
	result[id + 4 * offset] = H[4];
	result[id + 5 * offset] = H[5];
	result[id + 6 * offset] = H[6];
	result[id + 7 * offset] = H[7];
	result[id + 8 * offset] = H[8];


}


__global__ void cal_Homo_DLT(double* src, double* tar, double* result, int offset) {

	size_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= offset)
		return;
	//printf("%d\n", id);

	double cM_x = 0, cM_y = 0, cm_x = 0, cm_y = 0, sM_x = 0, sM_y = 0, sm_x = 0, sm_y = 0;
	cm_x += tar[id] + tar[id + 2 * offset] + tar[id + 4 * offset] + tar[id + 6 * offset];
	cm_x /= 4;

	sm_x += fabs(tar[id] - cm_x) + fabs(tar[id + 2 * offset] - cm_x) + fabs(tar[id + 4 * offset] - cm_x) + fabs(tar[id + 6 * offset] - cm_x);

	cm_y += tar[id + 1 * offset] + tar[id + 3 * offset] + tar[id + 5 * offset] + tar[id + 7 * offset];
	cm_y /= 4;

	sm_y += fabs(tar[id + 1 * offset] - cm_y) + fabs(tar[id + 3 * offset] - cm_y) + fabs(tar[id + 5 * offset] - cm_y) + fabs(tar[id + 7 * offset] - cm_y);


	cM_x += src[id] + src[id + 2 * offset] + src[id + 4 * offset] + src[id + 6 * offset];
	cM_x /= 4;
	sM_x = fabs(src[id] - cM_x) + fabs(src[id + 2 * offset] - cM_x) + fabs(src[id + 4 * offset] - cM_x) + fabs(src[id + 6 * offset] - cM_x);


	cM_y += src[id + 1 * offset] + src[id + 3 * offset] + src[id + 5 * offset] + src[id + 7 * offset];
	cM_y /= 4;
	sM_y = fabs(src[id + 1 * offset] - cM_y) + fabs(src[id + 3 * offset] - cM_y) + fabs(src[id + 5 * offset] - cM_y) + fabs(src[id + 7 * offset] - cM_y);


	sm_x = 4 / sm_x;
	sm_y = 4 / sm_y;
	sM_x = 4 / sM_x;
	sM_y = 4 / sM_y;


	double invHnorm[9] = { 0 }, Hnorm2[9] = { 0 };

	invHnorm[0] = 1. / sm_x;
	//invHnorm[id * 9 + 1] = 0;
	invHnorm[2] = cm_x;
	//invHnorm[id * 9 + 3] = 0;
	invHnorm[4] = 1. / sm_y;
	invHnorm[5] = cm_y;
	//invHnorm[id * 9 + 6] = 0;
	//invHnorm[id * 9 + 7] = 0;
	invHnorm[8] = 1;

	Hnorm2[0] = sM_x;
	//Hnorm2[id * 9 + 1] = 0;
	Hnorm2[2] = -cM_x * sM_x;
	//Hnorm2[id * 9 + 3] = 0;
	Hnorm2[4] = sM_y;
	Hnorm2[5] = -cM_y * sM_y;
	//Hnorm2[id * 9 + 6] = 0;
	//Hnorm2[id * 9 + 7] = 0;
	Hnorm2[8] = 1;


	double A[72] = { 0 };
	for (int i = 0; i < 4; i++) {
		int idx_x = 2 * i, idx_y = 2 * i + 1;
		double x = (tar[id + idx_x * offset] - cm_x) * sm_x, y = (tar[id + idx_y * offset] - cm_y) * sm_y;
		double X = (src[id + idx_x * offset] - cM_x) * sM_x, Y = (src[id + idx_y * offset] - cM_y) * sM_y;
		double Lx[] = { X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x };
		double Ly[] = { 0, 0, 0, X, Y, 1, -y * X, -y * Y, -y };

		for (int j = 0; j < 9; j++) {
			A[i * 18 + j] = Lx[j];
			A[(i * 2 + 1) * 9 + j] = Ly[j];
		}

	}


	int row = 8, col = 9;
	double diag[9], v[81], rvl[81];

	int _ = svd(A, diag, v, rvl, row, col);
	int index_diag = 0;
	for (int i = 1; i < 9; i++) {
		if (diag[i] < diag[index_diag])
			index_diag = i;
	}

	double _HO[9] = { 0 };
	for (int i = 0; i < 9; i++)
		_HO[i] = v[i * 9 + index_diag];


	double _Htemp[9] = { 0 };
	_Htemp[0] = invHnorm[0] * _HO[0] + invHnorm[1] * _HO[3] + invHnorm[2] * _HO[6];
	_Htemp[1] = invHnorm[0] * _HO[1] + invHnorm[1] * _HO[4] + invHnorm[2] * _HO[7];
	_Htemp[2] = invHnorm[0] * _HO[2] + invHnorm[1] * _HO[5] + invHnorm[2] * _HO[8];
	_Htemp[3] = invHnorm[3] * _HO[0] + invHnorm[4] * _HO[3] + invHnorm[5] * _HO[6];
	_Htemp[4] = invHnorm[3] * _HO[1] + invHnorm[4] * _HO[4] + invHnorm[5] * _HO[7];
	_Htemp[5] = invHnorm[3] * _HO[2] + invHnorm[4] * _HO[5] + invHnorm[5] * _HO[8];
	_Htemp[6] = invHnorm[6] * _HO[0] + invHnorm[7] * _HO[3] + invHnorm[8] * _HO[6];
	_Htemp[7] = invHnorm[6] * _HO[1] + invHnorm[7] * _HO[4] + invHnorm[8] * _HO[7];
	_Htemp[8] = invHnorm[6] * _HO[2] + invHnorm[7] * _HO[5] + invHnorm[8] * _HO[8];

	double res[9] = { 0 };
	res[0] = _Htemp[0] * Hnorm2[0] + _Htemp[1] * Hnorm2[3] + _Htemp[2] * Hnorm2[6];
	res[1] = _Htemp[0] * Hnorm2[1] + _Htemp[1] * Hnorm2[4] + _Htemp[2] * Hnorm2[7];
	res[2] = _Htemp[0] * Hnorm2[2] + _Htemp[1] * Hnorm2[5] + _Htemp[2] * Hnorm2[8];
	res[3] = _Htemp[3] * Hnorm2[0] + _Htemp[4] * Hnorm2[3] + _Htemp[5] * Hnorm2[6];
	res[4] = _Htemp[3] * Hnorm2[1] + _Htemp[4] * Hnorm2[4] + _Htemp[5] * Hnorm2[7];
	res[5] = _Htemp[3] * Hnorm2[2] + _Htemp[4] * Hnorm2[5] + _Htemp[5] * Hnorm2[8];
	res[6] = _Htemp[6] * Hnorm2[0] + _Htemp[7] * Hnorm2[3] + _Htemp[8] * Hnorm2[6];
	res[7] = _Htemp[6] * Hnorm2[1] + _Htemp[7] * Hnorm2[4] + _Htemp[8] * Hnorm2[7];
	res[8] = _Htemp[6] * Hnorm2[2] + _Htemp[7] * Hnorm2[5] + _Htemp[8] * Hnorm2[8];


	for (int i = 0; i < 9; i++)
		res[i] /= res[8];


	result[id] = res[0];
	result[id + 1 * offset] = res[1];
	result[id + 2 * offset] = res[2];
	result[id + 3 * offset] = res[3];
	result[id + 4 * offset] = res[4];
	result[id + 5 * offset] = res[5];
	result[id + 6 * offset] = res[6];
	result[id + 7 * offset] = res[7];
	result[id + 8 * offset] = res[8];

}





void cal_ACA(int numsofH, double* d_src_out, double* d_tar_out) {

	double* H_aka_d;
	double* H_aka_h = (double*)malloc(9 * numsofH * sizeof(double));
	CHECK(cudaMalloc((void**)&H_aka_d, 9 * numsofH * sizeof(double)));

	float elapsedTime_ref, elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 block_cal_aka = 32;
	dim3 grid_cal_aka = (numsofH + block_cal_aka.x - 1) / block_cal_aka.x;
	int loops = 1;
	

	cudaEventRecord(start, 0);
	for (int i = 0; i < loops; i++)
		cal_Homo_ACA << <grid_cal_aka, block_cal_aka >> > (d_src_out, d_tar_out, H_aka_d, numsofH);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime_ref, start, stop);
	loops = 10000 / elapsedTime_ref;


	cudaEventRecord(start, 0);
	for (int i = 0; i < loops; i++)
		cal_Homo_ACA << <grid_cal_aka, block_cal_aka >> > (d_src_out, d_tar_out, H_aka_d, numsofH);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("cal_Homo_ACA Execution configuration<<<%d,%d>>> Time elapsed %f ms\n\n", grid_cal_aka.x, block_cal_aka.x, elapsedTime/loops);
	CHECK(cudaMemcpy(H_aka_h, H_aka_d, 9 * numsofH * sizeof(double), cudaMemcpyDeviceToHost));
	cout << "----------------------------------------------------------" << endl;


	CHECK(cudaFree(H_aka_d));


}

void cal_SKS(int numsofH, double* d_src_out, double* d_tar_out) {
	double* H_SKS_d;
	double* H_SKS_h = (double*)malloc(9 * numsofH * sizeof(double));
	CHECK(cudaMalloc((void**)&H_SKS_d, 9 * numsofH * sizeof(double)));
	float elapsedTime, elapsedTime_ref;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	dim3 block_cal_sks = 32;
	dim3 grid_cal_sks = (numsofH + block_cal_sks.x - 1) / block_cal_sks.x;
	int loops = 1;

	cudaEventRecord(start, 0);
	for (int i = 0; i < loops; i++)
		cal_Homo_SKS << <grid_cal_sks, block_cal_sks >> > (d_src_out, d_tar_out, H_SKS_d, numsofH);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime_ref, start, stop);
	loops = 10000 / elapsedTime_ref;


	cudaEventRecord(start, 0);
	for (int i = 0; i < loops; i++)
		cal_Homo_SKS << <grid_cal_sks, block_cal_sks >> > (d_src_out, d_tar_out, H_SKS_d, numsofH);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("cal_Homo_SKS Execution configuration<<<%d,%d>>> Time elapsed %f ms\n\n", grid_cal_sks.x, block_cal_sks.x, elapsedTime / loops);
	CHECK(cudaMemcpy(H_SKS_h, H_SKS_d, 9 * numsofH * sizeof(double), cudaMemcpyDeviceToHost));
	cout << "----------------------------------------------------------" << endl;

	free(H_SKS_h);
	CHECK(cudaFree(H_SKS_d));

}

void cal_GPT(int numsofH, double* d_src_out, double* d_tar_out) {

	double* H_GPT_d;
	double* H_GPT_h = (double*)malloc(9 * numsofH * sizeof(double));
	CHECK(cudaMalloc((void**)&H_GPT_d, 9 * numsofH * sizeof(double)));

	float elapsedTime, elapsedTime_ref;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 block = 32;
	dim3 grid = (numsofH + block.x - 1) / block.x;
	int loops = 1;

	cudaEventRecord(start, 0);
	for (int i = 0; i < loops; i++)
		cal_Homo_GPT << <grid, block >> > (d_src_out, d_tar_out, H_GPT_d, numsofH);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime_ref, start, stop);
	loops = 10000 / elapsedTime_ref;

	cudaEventRecord(start, 0);
	for (int i = 0; i < loops; i++)
		cal_Homo_GPT << <grid, block >> > (d_src_out, d_tar_out, H_GPT_d, numsofH);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("cal_Homo_GPT Execution configuration<<<%d,%d>>> Time elapsed %f ms\n\n", grid.x, block.x, elapsedTime / loops);
	CHECK(cudaMemcpy(H_GPT_h, H_GPT_d, 9 * numsofH * sizeof(double), cudaMemcpyDeviceToHost));
	cout << "----------------------------------------------------------" << endl;

	free(H_GPT_h);
	CHECK(cudaFree(H_GPT_d));

}

void cal_GE(int numsofH, double* d_src_out, double* d_tar_out) {
	double* H_RHO_d;
	double* H_RHO_h = (double*)malloc(9 * numsofH * sizeof(double));
	CHECK(cudaMalloc((void**)&H_RHO_d, 9 * numsofH * sizeof(double)));

	float elapsedTime, elapsedTime_ref;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 block_cal_rho = 32;
	dim3 grid_cal_rho = (numsofH + block_cal_rho.x - 1) / block_cal_rho.x;
	int loops = 1;


	cudaEventRecord(start, 0);
	for (int i = 0; i < loops; i++)
		cal_Homo_GE << <grid_cal_rho, block_cal_rho >> > (d_src_out, d_tar_out, H_RHO_d, numsofH);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime_ref, start, stop);
	loops = 10000 / elapsedTime_ref;

	cudaEventRecord(start, 0);
	for (int i = 0; i < loops; i++)
		cal_Homo_GE << <grid_cal_rho, block_cal_rho >> > (d_src_out, d_tar_out, H_RHO_d, numsofH);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("cal_Homo_GE  Execution configuration<<<%d,%d>>> Time elapsed %f ms\n\n", grid_cal_rho.x, block_cal_rho.x, elapsedTime / loops);
	CHECK(cudaMemcpy(H_RHO_h, H_RHO_d, 9 * numsofH * sizeof(double), cudaMemcpyDeviceToHost));
	cout << "----------------------------------------------------------" << endl;

	free(H_RHO_h);
	CHECK(cudaFree(H_RHO_d));

}

void cal_HO(int numsofH, double* d_src_out, double* d_tar_out) {

	double* H_HO_d;
	double* H_HO_h = (double*)malloc(9 * numsofH * sizeof(double));
	CHECK(cudaMalloc((void**)&H_HO_d, 9 * numsofH * sizeof(double)));
	float elapsedTime, elapsedTime_ref;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 block_HO = 32;
	dim3 grid_HO = (numsofH + block_HO.x - 1) / block_HO.x;
	int loops = 1;

	cudaEventRecord(start, 0);
	for (int i = 0; i < loops; i++)
		cal_Homo_HO << <grid_HO, block_HO >> > (d_src_out, d_tar_out, H_HO_d, numsofH);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime_ref, start, stop);
	loops = 10000 / elapsedTime_ref;

	cudaEventRecord(start, 0);
	for (int i = 0; i < loops; i++)
		cal_Homo_HO << <grid_HO, block_HO >> > (d_src_out, d_tar_out, H_HO_d, numsofH);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("cal_Homo_HO  Execution configuration<<<%d,%d>>> Time elapsed %f ms\n\n", grid_HO.x, block_HO.x, elapsedTime / loops);
	CHECK(cudaMemcpy(H_HO_h, H_HO_d, 9 * numsofH * sizeof(double), cudaMemcpyDeviceToHost));
	cout << "----------------------------------------------------------" << endl;

	free(H_HO_h);
	CHECK(cudaFree(H_HO_d));

}

void cal_DLT(int numsofH, double* d_src_out, double* d_tar_out) {

	double* H_DLT_d;
	double* H_DLT_h = (double*)malloc(9 * numsofH * sizeof(double));
	CHECK(cudaMalloc((void**)&H_DLT_d, 9 * numsofH * sizeof(double)));
	float elapsedTime, elapsedTime_ref;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 block_DLT = 32;
	dim3 grid_DLT = (numsofH + block_DLT.x - 1) / block_DLT.x;
	int loops = 1;


	cudaEventRecord(start, 0);
	for (int i = 0; i < loops; i++)
		cal_Homo_DLT << <grid_DLT, block_DLT >> > (d_src_out, d_tar_out, H_DLT_d, numsofH);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime_ref, start, stop);
	loops = 10000 / elapsedTime_ref;

	cudaEventRecord(start, 0);
	for (int i = 0; i < loops; i++)
		cal_Homo_DLT << <grid_DLT, block_DLT >> > (d_src_out, d_tar_out, H_DLT_d, numsofH);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("cal_Homo_DLT Execution configuration<<<%d,%d>>> Time elapsed %f ms\n\n", grid_DLT.x, block_DLT.x, elapsedTime / loops);
	CHECK(cudaMemcpy(H_DLT_h, H_DLT_d, 9 * numsofH * sizeof(double), cudaMemcpyDeviceToHost));
	cout << "----------------------------------------------------------" << endl;

	free(H_DLT_h);
	CHECK(cudaFree(H_DLT_d));

}




int main(int argc, char** argv) {

	printf("Data preparing!!!\n");
	std::vector<cv::Point2f> pts_src;
	std::vector<cv::Point2f> pts_tar;
	read_points("orig_pts_wall.txt", pts_src, pts_tar);




	int N_p = pts_src.size();
	Point2d* p_src = (Point2d*)malloc(N_p * sizeof(Point2d)), * p_tar = (Point2d*)malloc(N_p * sizeof(Point2d));
	copy(pts_src.begin(), pts_src.end(), p_src);
	copy(pts_tar.begin(), pts_tar.end(), p_tar);

	// Number of homography matrices
	int numsOfH = 1<<12;

	if (argc == 2) {
		numsOfH = atoi(argv[1]);
	}
	cout << numsOfH << endl;


	double* h_src_out = (double*)malloc(8 * numsOfH * sizeof(double)), * h_tar_out = (double*)malloc(8 * numsOfH * sizeof(double));
	unsigned int* p_h = (unsigned int*)malloc(4 * numsOfH * sizeof(unsigned int));

	Point2d* d_src_in, * d_tar_in;
	double* d_src_out, * d_tar_out;
	unsigned int* p_d;

	CHECK(cudaMalloc((void**)&d_src_in, N_p * sizeof(Point2d)));
	CHECK(cudaMalloc((void**)&d_tar_in, N_p * sizeof(Point2d)));
	CHECK(cudaMalloc((void**)&p_d, 4 * numsOfH * sizeof(unsigned int)));
	CHECK(cudaMalloc((void**)&d_src_out, 8 * numsOfH * sizeof(double)));
	CHECK(cudaMalloc((void**)&d_tar_out, 8 * numsOfH * sizeof(double)));
	CHECK(cudaMemcpy(d_src_in, p_src, N_p * sizeof(Point2d), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_tar_in, p_tar, N_p * sizeof(Point2d), cudaMemcpyHostToDevice));


	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
	curandSetPseudoRandomGeneratorSeed(gen, 11ULL);
	curandGenerate(gen, p_d, 4 * numsOfH);
	

	dim3 block = 1024;
	dim3 grid = (numsOfH + block.x - 1) / block.x;
	get_rand_list << <grid, block >> > (p_d, N_p, d_src_in, d_tar_in, d_src_out, d_tar_out, numsOfH);





	//ACA----------------------------------------------------------------------
	cal_ACA(numsOfH, d_src_out, d_tar_out);
	//cout << "ACA COMPLETED" << endl;
	//-------------------------------------------------------------------------


	//SKS1----------------------------------------------------------------------
	cal_SKS(numsOfH, d_src_out, d_tar_out); 
	//cout << "SKS1 COMPLETED" << endl;
	//-------------------------------------------------------------------------


	//GE-----------------------------------------------------------------------
	cal_GE(numsOfH, d_src_out, d_tar_out);
	//cout << "RHO COMPLETED" << endl;
	//-------------------------------------------------------------------------


	//GPT----------------------------------------------------------------------
	cal_GPT(numsOfH, d_src_out, d_tar_out);
	//cout << "SKS2 COMPLETED" << endl;
	//-------------------------------------------------------------------------


	//HO---------------------------------------------------------------------
	cal_HO(numsOfH, d_src_out, d_tar_out);
	//cout << "HO COMPLETED" << endl;
	//------------------------------------------------------------------------


	//DLT---------------------------------------------------------------------
	cal_DLT(numsOfH, d_src_out, d_tar_out);
	//cout << "DLT COMPLETED" << endl;
	//------------------------------------------------------------------------


	

	curandDestroyGenerator(gen);
	cudaFree(d_src_in);
	cudaFree(d_tar_in);
	cudaFree(d_src_out);
	cudaFree(d_tar_out);
	cudaFree(p_d);
	free(p_h);
	free(h_src_out);
	free(h_tar_out);
	return 0;

}

