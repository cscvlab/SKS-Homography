#include <fstream>
#include "utils.h"

using namespace cv;

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
