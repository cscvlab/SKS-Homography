#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "DLT.hpp"
#include "SKS.hpp"
#include "GE.hpp"
#include "GPT.hpp"
#include "HO.hpp"

using namespace cv;

class TimeCalculator
{
	long long startTick;
public:
	TimeCalculator()
	{
		startTick = 0;
	}
	void start()
	{
		startTick = cv::getTickCount();
	}
	double end() const
	{
		return (cv::getTickCount() - startTick) / (static_cast<double>(cv::getTickFrequency()));
	}
};

void read_points(std::string filename, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);


