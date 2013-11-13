#include<iostream>
#include<fstream>
#include<vector>
#include<opencv2/features2d/features2d.hpp>
#include<opencv/highgui.h>
using namespace std;
using namespace cv;

int main() {
	ifstream fin("D:/project/action/sample_data/keypoints", ifstream::in);
	vector<KeyPoint> keypoints;
	while(!fin.eof()) {
		KeyPoint kp;
		fin >> kp.pt.x >> kp.pt.y >> kp.size;
		keypoints.push_back(kp);
	}
	fin.close();
	
	Mat diff_img = imread("D:/project/action/sample_data/diff_img.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat descriptors;
	FREAK extractor;
	extractor.compute(diff_img, keypoints, descriptors);

	ofstream fout("D:/project/action/sample_data/descriptors_test");
	for(int y=0; y<descriptors.rows; ++y) {
		for(int x=0; x<descriptors.cols; ++x) {
			fout << (int)descriptors.at<uchar>(y,x) << " ";
		}
		fout << endl;
	}
	fout.close();

	/*
	cout << "Keypoints.size: " << keypoints.size() << endl;
	for(auto it=keypoints.begin(); it!= keypoints.end(); ++it)
		cout << it->pt.x << " " << it->pt.y << " " << it->size << endl;
	int k;
	cin >> k;
	*/
	return 0;
}