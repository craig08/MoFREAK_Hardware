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
	const int FREAK_DEF_PAIRS[NB_PAIRS] =
{ // default pairs
     404,431,818,511,181,52,311,874,774,543,719,230,417,205,11,
     560,149,265,39,306,165,857,250,8,61,15,55,717,44,412,
     592,134,761,695,660,782,625,487,549,516,271,665,762,392,178,
     796,773,31,672,845,548,794,677,654,241,831,225,238,849,83,
     691,484,826,707,122,517,583,731,328,339,571,475,394,472,580,
     381,137,93,380,327,619,729,808,218,213,459,141,806,341,95,
     382,568,124,750,193,749,706,843,79,199,317,329,768,198,100,
     466,613,78,562,783,689,136,838,94,142,164,679,219,419,366,
     418,423,77,89,523,259,683,312,555,20,470,684,123,458,453,833,
     72,113,253,108,313,25,153,648,411,607,618,128,305,232,301,84,
     56,264,371,46,407,360,38,99,176,710,114,578,66,372,653,
     129,359,424,159,821,10,323,393,5,340,891,9,790,47,0,175,346,
     236,26,172,147,574,561,32,294,429,724,755,398,787,288,299,
     769,565,767,722,757,224,465,723,498,467,235,127,802,446,233,
     544,482,800,318,16,532,801,441,554,173,60,530,713,469,30,
     212,630,899,170,266,799,88,49,512,399,23,500,107,524,90,
     194,143,135,192,206,345,148,71,119,101,563,870,158,254,214,
     276,464,332,725,188,385,24,476,40,231,620,171,258,67,109,
     844,244,187,388,701,690,50,7,850,479,48,522,22,154,12,659,
     736,655,577,737,830,811,174,21,237,335,353,234,53,270,62,
     182,45,177,245,812,673,355,556,612,166,204,54,248,365,226,
     242,452,700,685,573,14,842,481,468,781,564,416,179,405,35,
     819,608,624,367,98,643,448,2,460,676,440,240,130,146,184,
     185,430,65,807,377,82,121,708,239,310,138,596,730,575,477,
     851,797,247,27,85,586,307,779,326,494,856,324,827,96,748,
     13,397,125,688,702,92,293,716,277,140,112,4,80,855,839,1,
     413,347,584,493,289,696,19,751,379,76,73,115,6,590,183,734,
     197,483,217,344,330,400,186,243,587,220,780,200,793,246,824,
     41,735,579,81,703,322,760,720,139,480,490,91,814,813,163,
     152,488,763,263,425,410,576,120,319,668,150,160,302,491,515,
     260,145,428,97,251,395,272,252,18,106,358,854,485,144,550,
     131,133,378,68,102,104,58,361,275,209,697,582,338,742,589,
     325,408,229,28,304,191,189,110,126,486,211,547,533,70,215,
     670,249,36,581,389,605,331,518,442,822
};

    if( image.empty() )
        return;
    if( keypoints.empty() )
        return;

	// buildPattern
	vector<PatternPoint> patternLookup;
	int patternSizes[NB_SCALES];
    DescriptionPair descriptionPairs[NB_PAIRS];
    OrientationPair orientationPairs[NB_ORIENPAIRS];

	patternLookup.resize(NB_SCALES*FREAK_NB_ORIENTATION*FREAK_NB_POINTS);
    double scaleStep = pow(2.0, (double)(nOctaves)/NB_SCALES ); // 2 ^ ( (nOctaves-1) /nbScales)
    double scalingFactor, alpha, beta, theta = 0;

    // pattern definition, radius normalized to 1.0 (outer point position+sigma=1.0)
    const int n[8] = {6,6,6,6,6,6,6,1}; // number of points on each concentric circle (from outer to inner)
    const double bigR(2.0/3.0); // bigger radius
    const double smallR(2.0/24.0); // smaller radius
    const double unitSpace( (bigR-smallR)/21.0 ); // define spaces between concentric circles (from center to outer: 1,2,3,4,5,6)
    // radii of the concentric cirles (from outer to inner)
    const double radius[8] = {bigR, bigR-6*unitSpace, bigR-11*unitSpace, bigR-15*unitSpace, bigR-18*unitSpace, bigR-20*unitSpace, smallR, 0.0};
    // sigma of pattern points (each group of 6 points on a concentric cirle has the same sigma)
    const double sigma[8] = {radius[0]/2.0, radius[1]/2.0, radius[2]/2.0,
                             radius[3]/2.0, radius[4]/2.0, radius[5]/2.0,
                             radius[6]/2.0, radius[6]/2.0
                            };
    // fill the lookup table
    for( int scaleIdx=0; scaleIdx < NB_SCALES; ++scaleIdx ) {
        patternSizes[scaleIdx] = 0; // proper initialization
        scalingFactor = pow(scaleStep,scaleIdx); //scale of the pattern, scaleStep ^ scaleIdx

        for( int orientationIdx = 0; orientationIdx < FREAK_NB_ORIENTATION; ++orientationIdx ) {
            theta = double(orientationIdx)* 2*CV_PI/double(FREAK_NB_ORIENTATION); // orientation of the pattern
            int pointIdx = 0;

            PatternPoint* patternLookupPtr = &patternLookup[0];
            for( size_t i = 0; i < 8; ++i ) {
                for( int k = 0 ; k < n[i]; ++k ) {
                    beta = CV_PI/n[i] * (i%2); // orientation offset so that groups of points on each circles are staggered
                    alpha = double(k)* 2*CV_PI/double(n[i])+beta+theta;

                    // add the point to the look-up table
                    PatternPoint& point = patternLookupPtr[ scaleIdx*FREAK_NB_ORIENTATION*FREAK_NB_POINTS+orientationIdx*FREAK_NB_POINTS+pointIdx ];
                    point.x = static_cast<float>(radius[i] * cos(alpha) * scalingFactor * patternScale);
                    point.y = static_cast<float>(radius[i] * sin(alpha) * scalingFactor * patternScale);
                    point.sigma = static_cast<float>(sigma[i] * scalingFactor * patternScale);

                    // adapt the sizeList if necessary
                    const int sizeMax = static_cast<int>(ceil((radius[i]+sigma[i])*scalingFactor*patternScale)) + 1;
                    if( patternSizes[scaleIdx] < sizeMax )
                        patternSizes[scaleIdx] = sizeMax;

                    ++pointIdx;
                }
            }
        }
    }

    // build the list of orientation pairs
    orientationPairs[0].i=0; orientationPairs[0].j=3; orientationPairs[1].i=1; orientationPairs[1].j=4; orientationPairs[2].i=2; orientationPairs[2].j=5;
    orientationPairs[3].i=0; orientationPairs[3].j=2; orientationPairs[4].i=1; orientationPairs[4].j=3; orientationPairs[5].i=2; orientationPairs[5].j=4;
    orientationPairs[6].i=3; orientationPairs[6].j=5; orientationPairs[7].i=4; orientationPairs[7].j=0; orientationPairs[8].i=5; orientationPairs[8].j=1;

    orientationPairs[9].i=6; orientationPairs[9].j=9; orientationPairs[10].i=7; orientationPairs[10].j=10; orientationPairs[11].i=8; orientationPairs[11].j=11;
    orientationPairs[12].i=6; orientationPairs[12].j=8; orientationPairs[13].i=7; orientationPairs[13].j=9; orientationPairs[14].i=8; orientationPairs[14].j=10;
    orientationPairs[15].i=9; orientationPairs[15].j=11; orientationPairs[16].i=10; orientationPairs[16].j=6; orientationPairs[17].i=11; orientationPairs[17].j=7;

    orientationPairs[18].i=12; orientationPairs[18].j=15; orientationPairs[19].i=13; orientationPairs[19].j=16; orientationPairs[20].i=14; orientationPairs[20].j=17;
    orientationPairs[21].i=12; orientationPairs[21].j=14; orientationPairs[22].i=13; orientationPairs[22].j=15; orientationPairs[23].i=14; orientationPairs[23].j=16;
    orientationPairs[24].i=15; orientationPairs[24].j=17; orientationPairs[25].i=16; orientationPairs[25].j=12; orientationPairs[26].i=17; orientationPairs[26].j=13;

    orientationPairs[27].i=18; orientationPairs[27].j=21; orientationPairs[28].i=19; orientationPairs[28].j=22; orientationPairs[29].i=20; orientationPairs[29].j=23;
    orientationPairs[30].i=18; orientationPairs[30].j=20; orientationPairs[31].i=19; orientationPairs[31].j=21; orientationPairs[32].i=20; orientationPairs[32].j=22;
    orientationPairs[33].i=21; orientationPairs[33].j=23; orientationPairs[34].i=22; orientationPairs[34].j=18; orientationPairs[35].i=23; orientationPairs[35].j=19;

    orientationPairs[36].i=24; orientationPairs[36].j=27; orientationPairs[37].i=25; orientationPairs[37].j=28; orientationPairs[38].i=26; orientationPairs[38].j=29;
    orientationPairs[39].i=30; orientationPairs[39].j=33; orientationPairs[40].i=31; orientationPairs[40].j=34; orientationPairs[41].i=32; orientationPairs[41].j=35;
    orientationPairs[42].i=36; orientationPairs[42].j=39; orientationPairs[43].i=37; orientationPairs[43].j=40; orientationPairs[44].i=38; orientationPairs[44].j=41;

    for( unsigned m = NB_ORIENPAIRS; m--; ) {
        const float dx = patternLookup[orientationPairs[m].i].x-patternLookup[orientationPairs[m].j].x;
        const float dy = patternLookup[orientationPairs[m].i].y-patternLookup[orientationPairs[m].j].y;
        const float norm_sq = (dx*dx+dy*dy);
        orientationPairs[m].weight_dx = int((dx/(norm_sq))*4096.0+0.5);
        orientationPairs[m].weight_dy = int((dy/(norm_sq))*4096.0+0.5);
    }

    // build the list of description pairs
    std::vector<DescriptionPair> allPairs;
    for( unsigned int i = 1; i < (unsigned int)FREAK_NB_POINTS; ++i ) {
        // (generate all the pairs)
        for( unsigned int j = 0; (unsigned int)j < i; ++j ) {
            DescriptionPair pair = {(uchar)i,(uchar)j};
            allPairs.push_back(pair);
        }
    }
    for( int i = 0; i < NB_PAIRS; ++i )
            descriptionPairs[i] = allPairs[FREAK_DEF_PAIRS[i]];


    Mat imgIntegral;
    integral(image, imgIntegral);
    std::vector<int> kpScaleIdx(keypoints.size()); // used to save pattern scale index corresponding to each keypoints
    const std::vector<int>::iterator ScaleIdxBegin = kpScaleIdx.begin(); // used in std::vector erase function
    const std::vector<cv::KeyPoint>::iterator kpBegin = keypoints.begin(); // used in std::vector erase function
    const float sizeCst = static_cast<float>(NB_SCALES/(FREAK_LOG2* nOctaves));
    uchar pointsValue[FREAK_NB_POINTS];
    int thetaIdx = 0;
    int direction0;
    int direction1;

    // compute the scale index corresponding to the keypoint size and remove keypoints close to the border
        for( size_t k = keypoints.size(); k--; ) {
            //Is k non-zero? If so, decrement it and continue"
            kpScaleIdx[k] = max( (int)(log(keypoints[k].size/FREAK_SMALLEST_KP_SIZE)*sizeCst+0.5) ,0);
            if( kpScaleIdx[k] >= NB_SCALES )
                kpScaleIdx[k] = NB_SCALES-1;

            if( keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] || //check if the description at this specific position and scale fits inside the image
                 keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
                 keypoints[k].pt.x >= image.cols-patternSizes[kpScaleIdx[k]] ||
                 keypoints[k].pt.y >= image.rows-patternSizes[kpScaleIdx[k]]
               ) {
                keypoints.erase(kpBegin+k);
                kpScaleIdx.erase(ScaleIdxBegin+k);
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

            // extracting descriptor preserving the order of SSE version
            int cnt = 0;
            for( int n = 7; n < NB_PAIRS; n += 128)
            {
                for( int m = 8; m--; )
                {
                    int nm = n-m;
                    for(int kk = nm+15*8; kk >= nm; kk-=8, ++cnt)
                    {
                        ptr->set(kk, pointsValue[descriptionPairs[cnt].i] >= pointsValue[descriptionPairs[cnt].j]);
                    }
                }
            }
            --ptr;
        }
    
}


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
	//Mat current_frame = imread("D:/project/action/sample_data/current_frame.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat prev_frame = imread("D:/project/action/sample_data/prev_frame.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat descriptors;
	//FREAK extractor;
	//extractor.compute(diff_img, keypoints, descriptors);
	computeImpl(diff_img, keypoints, descriptors);
	/* test MIP
	for(auto keypt=keypoints.begin(); keypt!=keypoints.end(); ++keypt) {
		unsigned int MIP = motionInterchangePattern(current_frame, prev_frame, keypt->pt.x, keypt->pt.y);
		cout << MIP << endl;
	}
	*/
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
		*/
	//int k;
	//cin >> k;
	
	return 0;
}