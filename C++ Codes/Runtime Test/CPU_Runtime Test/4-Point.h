#pragma once
#include "DLT.hpp"
#include "SKS.hpp"
#include "GE.hpp"
#include "GPT.hpp"
#include "HO.hpp"
#include "utils.h"

class Solve_4Points {
	double DLT_time, SKS_time, GE_time, GPT_time, ACA_time, HO_time;
	double SKS_double_time, ACA_double_time;
	long long int num_randomSet = 100000000;

public:
	void run(std::vector<cv::Point2f>, std::vector<cv::Point2f>);
};
