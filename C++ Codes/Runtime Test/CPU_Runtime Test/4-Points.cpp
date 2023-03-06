#include "4-Point.h"
#include <iostream>

using namespace cv;

void Solve_4Points::run(std::vector<cv::Point2f> pts_src, std::vector<cv::Point2f> pts_tar) {
	int N_p = pts_src.size();
	std::vector<std::vector<Point2f>> test_points1, test_points2;
	RNG rng(static_cast<unsigned>(time(nullptr)));

	TimeCalculator timeCalculator;

	for (int i = 0; i < max(static_cast<long long>(1), num_randomSet / 10000); i++) {
		std::vector<Point2f> tmp_points1, tmp_points2;
		for (int j = 0; j < 4; j++) {
			int index = rng.uniform(0, N_p);
			tmp_points1.push_back(pts_src[index]);
			tmp_points2.push_back(pts_tar[index]);
		}
		test_points1.push_back(tmp_points1);
		test_points2.push_back(tmp_points2);
	}

	int cycleTimes = num_randomSet;
	Mat DLT_H(3, 3, CV_64FC1);
	Mat GPT_H(3, 3, CV_64FC1);
	Mat GE_H(3, 3, CV_64FC1);
	Mat SKS_H(3, 3, CV_64FC1);
	Mat ACA_H(3, 3, CV_64FC1);
	Matx33d HO_H;

	// randomly select one set of 4 points to compute homography
	int ind = rng.uniform(0, max(static_cast<long long>(1), num_randomSet / 10000));
	std::vector<Point2f> pts_4_src = test_points1[ind];
	std::vector<Point2f> pts_4_tar = test_points2[ind];

	float src_list[8], tar_list[8];
	for (int i = 0; i < 4; i++)
	{
		src_list[i * 2] = pts_4_src[i].x;
		src_list[i * 2 + 1] = pts_4_src[i].y;
		tar_list[i * 2] = pts_4_tar[i].x;
		tar_list[i * 2 + 1] = pts_4_tar[i].y;
	}

	timeCalculator.start();
	for (int k = 0; k < cycleTimes; k++) {
		runKernel_DLT(pts_4_src, pts_4_tar, DLT_H);
	}
	DLT_time = timeCalculator.end();


	timeCalculator.start();
	for (int k = 0; k < cycleTimes; k++) {
		runKernel_getPerspectiveTransform(pts_4_src, pts_4_tar, GPT_H);
	}
	GPT_time = timeCalculator.end();


	timeCalculator.start();
	for (int k = 0; k < cycleTimes; k++) {
		my_ho::my_homographyHO(pts_4_src, pts_4_tar, HO_H);
	}
	HO_time = timeCalculator.end();

	float H_GE[9];
	timeCalculator.start();
	for (int k = 0; k < cycleTimes; k++) {
		runKernelGE_Optimized(src_list, tar_list, H_GE);
	}
	GE_time = timeCalculator.end();

	float H_ACA[9];
	timeCalculator.start();
	for (int k = 0; k < cycleTimes; k++) {
		sks::runKernel_ACA(src_list, tar_list, H_ACA);
	}
	ACA_time = timeCalculator.end();

	double H_ACA_double[9];
	timeCalculator.start();
	for (int k = 0; k < cycleTimes; k++) {
		sks::runKernel_ACA_double(src_list, tar_list, H_ACA_double);
	}
	ACA_double_time = timeCalculator.end();

	float H_SKS[9];
	timeCalculator.start();
	for (int k = 0; k < cycleTimes; k++) {
		sks::runKernel_SKS(src_list, tar_list, H_SKS);
	}
	SKS_time = timeCalculator.end();


	double H_SKS_double[9];
	timeCalculator.start();
	for (int k = 0; k < cycleTimes; k++) {
		sks::runKernel_SKS_double(src_list, tar_list, H_SKS_double);
	}
	SKS_double_time = timeCalculator.end();

	double microsecond_multiply = 1e6;

	std::cout << std::fixed << "4 points' mean runtime (us) of " << cycleTimes << " times:" << std::endl;

	std::cout << std::fixed << "GE_runtime = " << GE_time * microsecond_multiply / cycleTimes << std::endl;
	std::cout << std::fixed << "DLT_runtime = " << DLT_time * microsecond_multiply / cycleTimes << std::endl;
	std::cout << std::fixed << "GPT_runtime = " << GPT_time * microsecond_multiply / cycleTimes << std::endl;
	std::cout << std::fixed << "HO_runtime  = " << HO_time * microsecond_multiply / cycleTimes << std::endl;
	std::cout << std::fixed << "SKS_runtime = " << SKS_time * microsecond_multiply / cycleTimes << std::endl;
	std::cout << std::fixed << "SKS_double_runtime = " << SKS_double_time * microsecond_multiply / cycleTimes << std::endl;
	std::cout << std::fixed << "ACA_runtime = " << ACA_time * microsecond_multiply / cycleTimes << std::endl;
	std::cout << std::fixed << "ACA_double_runtime = " << ACA_double_time * microsecond_multiply / cycleTimes << std::endl;

}

