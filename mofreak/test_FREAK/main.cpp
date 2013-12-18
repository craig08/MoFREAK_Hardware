#include<iostream>
#include<fstream>
#include<vector>
#include<bitset>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv/highgui.h>
#include<opencv/cxcore.h>
using namespace std;
using namespace cv;

struct PatternPoint
{
    float x; // x coordinate relative to center
    float y; // x coordinate relative to center
    float sigma; // Gaussian smoothing sigma
};

struct DescriptionPair
{
    uchar i; // index of the first point
    uchar j; // index of the second point
};

struct OrientationPair
{
    uchar i; // index of the first point
    uchar j; // index of the second point
    int weight_dx; // dx/(norm_sq))*4096
    int weight_dy; // dy/(norm_sq))*4096
};

/*
unsigned int motionInterchangePattern(cv::Mat &current_frame, cv::Mat &prev_frame, int x, int y)
{
	const int THETA = 288;//10368;//5184;//2592;//1296; // 288 means an average intensity difference of at least 32 per pixel.
	// extract patch on current frame.
	cv::Rect roi(x - 1, y - 1, 3, 3);
	cv::Mat patch_t(current_frame, roi);

	// extract patches from previous frame.
	vector<cv::Mat> previous_patches;
	// (-4, 0)
	previous_patches.push_back(prev_frame(cv::Rect((x - 4) - 1, y - 1, 3, 3)));
	// (-3, 3)
	previous_patches.push_back(prev_frame(cv::Rect((x - 3) - 1, (y + 3) - 1, 3, 3)));
	// (0, 4)
	previous_patches.push_back(prev_frame(cv::Rect(x - 1, (y + 4) - 1, 3, 3)));
	// (3, 3)
	previous_patches.push_back(prev_frame(cv::Rect((x + 3) - 1, (y + 3) - 1, 3, 3)));
	// (4, 0)
	previous_patches.push_back(prev_frame(cv::Rect((x + 4) - 1, y - 1, 3, 3)));
	// (3, -3)
	previous_patches.push_back(prev_frame(cv::Rect((x + 3) - 1, (y - 3) - 1, 3, 3)));
	// (0, -4)
	previous_patches.push_back(prev_frame(cv::Rect(x - 1, (y - 4) - 1, 3, 3)));
	// (-3, -3)
	previous_patches.push_back(prev_frame(cv::Rect((x - 3) - 1, (y - 3) - 1, 3, 3)));

	// now do SSD between current patch and all of those patches.
	// opencv might have an optimized ssd, i didn't find it though.
	unsigned int bit = 1;
	unsigned int descriptor = 0;
	for (auto it = previous_patches.begin(); it != previous_patches.end(); ++it)
	{
		int ssd = 0;
		uchar *p = patch_t.data;
		uchar *p2 = it->data;
		for (int row = 0; row < 3; ++row)
		{
			for (int col = 0; col < 3; ++col)
			{
				ssd += (int)pow((float)((*p) - (*p2)), 2);
				p++;
				p2++;
			}
		}

		if (ssd > THETA) 
		{
			descriptor |= bit;
		}
		bit <<= 1;
	}

	return descriptor;
}
*/

uchar meanIntensity( const cv::Mat& image, const cv::Mat& integral,
                            const float kp_x,
                            const float kp_y,
                            const unsigned int scale,
                            const unsigned int rot,
                            const unsigned int point,
							const vector<PatternPoint>& patternLookup) {
    // get point position in image
    const int FREAK_NB_ORIENTATION = 256;
    const int FREAK_NB_POINTS = 43;
    const PatternPoint& FreakPoint = patternLookup[scale*FREAK_NB_ORIENTATION*FREAK_NB_POINTS + rot*FREAK_NB_POINTS + point];
    const float xf = FreakPoint.x+kp_x;
    const float yf = FreakPoint.y+kp_y;
    const int x = int(xf);
    const int y = int(yf);
    const int& imagecols = image.cols;

    // get the sigma:
    const float radius = FreakPoint.sigma;

    // calculate output:
    if( radius < 0.5 ) {
        // interpolation multipliers:
        const int r_x = static_cast<int>((xf-x)*1024);
        const int r_y = static_cast<int>((yf-y)*1024);
        const int r_x_1 = (1024-r_x);
        const int r_y_1 = (1024-r_y);
        uchar* ptr = image.data+x+y*imagecols;
        unsigned int ret_val;
        // linear interpolation:
        ret_val = (r_x_1*r_y_1*int(*ptr));
        ptr++;
        ret_val += (r_x*r_y_1*int(*ptr));
        ptr += imagecols;
        ret_val += (r_x*r_y*int(*ptr));
        ptr--;
        ret_val += (r_x_1*r_y*int(*ptr));
        //return the rounded mean
        ret_val += 2 * 1024 * 1024;
        return static_cast<uchar>(ret_val / (4 * 1024 * 1024));
    }

    // expected case:

    // calculate borders
    const int x_left = int(xf-radius+0.5);
    const int y_top = int(yf-radius+0.5);
    const int x_right = int(xf+radius+1.5);//integral image is 1px wider
    const int y_bottom = int(yf+radius+1.5);//integral image is 1px higher
    int ret_val;

    ret_val = integral.at<int>(y_bottom,x_right);//bottom right corner
    ret_val -= integral.at<int>(y_bottom,x_left);
    ret_val += integral.at<int>(y_top,x_left);
    ret_val -= integral.at<int>(y_top,x_right);
    ret_val = ret_val/( (x_right-x_left)* (y_bottom-y_top) );
    //~ std::cout<<integral.step[1]<<std::endl;
    return static_cast<uchar>(ret_val);
}

void computeImpl( const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors )  {
    // NB_SCALES = 64, NB_PAIRS = 512, NB_ORIENPAIRS = 45
	const int NB_SCALES = 64;
	const int NB_PAIRS = 512;
	const int NB_ORIENPAIRS = 45;
    const double FREAK_LOG2 = 0.693147180559945;
    const int nOctaves = 4;
    const int FREAK_NB_ORIENTATION = 256;
    const int FREAK_NB_POINTS = 43;
    const int FREAK_SMALLEST_KP_SIZE = 7; // smallest size of keypoints
	const float patternScale = 22.0f;
    if( image.empty() )
        return;
    if( keypoints.empty() )
        return;
	// buildPattern
	vector<PatternPoint> patternLookup;
	int patternSizes[NB_SCALES];
    DescriptionPair descriptionPairs[NB_PAIRS];
    DescriptionPair newDesPairs[NB_PAIRS];
    OrientationPair orientationPairs[NB_ORIENPAIRS];
    // Read pattern
	ifstream fin("D:/project/action/sample_data/patternLookup");
    while(!fin.eof()) {
        PatternPoint temp;
        fin >> temp.x >> temp.y >> temp.sigma;
        patternLookup.push_back(temp);
    }
    fin.close();
    fin.open("D:/project/action/sample_data/patternSizes");
    int idx = 0;
    while(!fin.eof())
        fin >> patternSizes[idx++];
    fin.close();
    fin.open("D:/project/action/sample_data/orientationPairs");
    idx = 0;
    while(!fin.eof()) {
        int i, j;
        fin >> i >> j >> orientationPairs[idx].weight_dx >> orientationPairs[idx].weight_dy;
        orientationPairs[idx].i = i; 
        orientationPairs[idx].j = j;
        if(idx == NB_ORIENPAIRS-1) break;
        else ++idx;
    }
    fin.close();
    fin.open("D:/project/action/sample_data/newDesPairs");
    idx = 0;
    while(!fin.eof()) {
        int i, j;
        fin >> i >> j;
        newDesPairs[idx].i = i;
        newDesPairs[idx].j = j;
        if(idx == NB_PAIRS-1) break;
        else ++idx;
    }
    fin.close();     
	// end building pattern    
    
    Mat imgIntegral;
    integral(image, imgIntegral);
    std::vector<int> kpScaleIdx(keypoints.size()); // used to save pattern scale index corresponding to each keypoints
    const std::vector<int>::iterator ScaleIdxBegin = kpScaleIdx.begin(); 
    const std::vector<cv::KeyPoint>::iterator kpBegin = keypoints.begin(); 
    const float sizeCst = static_cast<float>(NB_SCALES/(FREAK_LOG2* nOctaves));
    uchar pointsValue[FREAK_NB_POINTS];
    int thetaIdx = 0;
    int direction0;
    int direction1;

    // compute the scale index corresponding to the keypoint size and remove keypoints close to the border
    for( size_t k = keypoints.size(); k--; ) {
        kpScaleIdx[k] = max( (int)(log(keypoints[k].size/FREAK_SMALLEST_KP_SIZE)*sizeCst+0.5) ,0);
        if( kpScaleIdx[k] >= NB_SCALES )
            kpScaleIdx[k] = NB_SCALES-1;
        //cout << "keypoints.size: " << keypoints[k].size << " kpScaleIdx: " << kpScaleIdx[k] << endl;
        //cout << "x: " << keypoints[k].pt.x << " y: " << keypoints[k].pt.y << " patternSizes[kpScaleIdx[k]]: " << patternSizes[kpScaleIdx[k]] << endl;
        //check if the description at this specific position and scale fits inside the image
        if( keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] || 
            keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
            keypoints[k].pt.x >= image.cols-patternSizes[kpScaleIdx[k]] ||
            keypoints[k].pt.y >= image.rows-patternSizes[kpScaleIdx[k]]
           ) {
            keypoints.erase(kpBegin+k);
            kpScaleIdx.erase(ScaleIdxBegin+k);
            //cout << "Erased!" << endl << endl;
        }
    }

    // allocate descriptor memory, estimate orientations, extract descriptors
    // extract the best comparisons only
    descriptors = cv::Mat::zeros((int)keypoints.size(), NB_PAIRS/8, CV_8U);
    std::bitset<NB_PAIRS>* ptr = (std::bitset<NB_PAIRS>*) (descriptors.data+(keypoints.size()-1)*descriptors.step[0]);
    for( size_t k = keypoints.size(); k--; ) {
        // estimate orientation (gradient)
        // get the points intensity value in the un-rotated pattern
        for( int i = FREAK_NB_POINTS; i--; ) {
            pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x,keypoints[k].pt.y, kpScaleIdx[k], 0, i, patternLookup);
        }
        direction0 = 0;
        direction1 = 0;
        for( int m = 45; m--; ) {
            //iterate through the orientation pairs
            const int delta = (pointsValue[ orientationPairs[m].i ]-pointsValue[ orientationPairs[m].j ]);
            direction0 += delta*(orientationPairs[m].weight_dx)/2048;
            direction1 += delta*(orientationPairs[m].weight_dy)/2048;
        }

        keypoints[k].angle = static_cast<float>(atan2((float)direction1,(float)direction0)*(180.0/CV_PI));//estimate orientation
        thetaIdx = int(FREAK_NB_ORIENTATION*keypoints[k].angle*(1/360.0)+0.5);
        if( thetaIdx < 0 )
            thetaIdx += FREAK_NB_ORIENTATION;

        if( thetaIdx >= FREAK_NB_ORIENTATION )
            thetaIdx -= FREAK_NB_ORIENTATION;
            
        // extract descriptor at the computed orientation
        for( int i = FREAK_NB_POINTS; i--; ) {
            pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x,keypoints[k].pt.y, kpScaleIdx[k], thetaIdx, i, patternLookup);
        }
        // extracting descriptor
        for(int n = 0; n < NB_PAIRS; ++n)
            ptr->set(n, pointsValue[newDesPairs[n].i] >= pointsValue[newDesPairs[n].j]);
        --ptr;
    }
    
}


int main() {
	ifstream fin("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/keypoints", ifstream::in);
	vector<KeyPoint> keypoints;
	while(!fin.eof()) {
		KeyPoint kp;
		fin >> kp.pt.x >> kp.pt.y >> kp.size;
		keypoints.push_back(kp);
	}
	fin.close();
	
	Mat diff_img = imread("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/diff_img.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat draw;
	//drawKeypoints(diff_img, keypoints, draw, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	//imwrite("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/draw_initial.png", draw);
	//Mat current_frame = imread("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/current_frame.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat prev_frame = imread("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/prev_frame.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat descriptors;
	FREAK extractor;
	extractor.compute(diff_img, keypoints, descriptors);
	//computeImpl(diff_img, keypoints, descriptors);
	/* test MIP
	for(auto keypt=keypoints.begin(); keypt!=keypoints.end(); ++keypt) {
		unsigned int MIP = motionInterchangePattern(current_frame, prev_frame, keypt->pt.x, keypt->pt.y);
		cout << MIP << endl;
	}
	*/
	ofstream fout("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/keypoints_final");
	for(auto y=keypoints.begin(); y!=keypoints.end(); ++y) {
		fout << y->pt.x << " " << y->pt.y << " " << y->size << " " << y->angle << endl;
	}
	fout.close();	
	fout.open("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/descriptors_test");
	for(int y=0; y<descriptors.rows; ++y) {
		for(int x=0; x<descriptors.cols; ++x) {
			fout << (int)descriptors.at<uchar>(y,x) << " ";
		}
		fout << endl;
	}
	fout.close();	
	//drawKeypoints(diff_img, keypoints, draw, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	//imwrite("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/draw_after_extraction.png", draw);

	//cvWaitKey(0);
	system("pause");
	
	return 0;
}