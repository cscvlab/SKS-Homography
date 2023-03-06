#include "utils.h"
#include "4-Point.h"
#include <iostream>

int main() {

	std::vector<cv::Point2f> pts_src;
	std::vector<cv::Point2f> pts_tar;
	read_points("orig_pts_wall.txt", pts_src, pts_tar);

	Solve_4Points step1;
	step1.run(pts_src, pts_tar);
}
