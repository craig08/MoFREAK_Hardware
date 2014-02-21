#include<iostream>
#include<fstream>
#include<vector>
#include<bitset>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/features2d.hpp>
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

struct SurfHF
{
    int p0, p1, p2, p3;
    float w;

    SurfHF(): p0(0), p1(0), p2(0), p3(0), w(0) {}
};

inline float calcHaarPattern( const int* origin, const SurfHF* f, int n )
{
    double d = 0;
    for( int k = 0; k < n; k++ )
        d += (origin[f[k].p0] + origin[f[k].p3] - origin[f[k].p1] - origin[f[k].p2])*f[k].w;
    return (float)d;
}

void
resizeHaarPattern( const int src[][5], SurfHF* dst, int n, int oldSize, int newSize, int widthStep )
{
    float ratio = (float)newSize/oldSize;
    for( int k = 0; k < n; k++ )
    {
        int dx1 = cvRound( ratio*src[k][0] );
        int dy1 = cvRound( ratio*src[k][1] );
        int dx2 = cvRound( ratio*src[k][2] );
        int dy2 = cvRound( ratio*src[k][3] );
        dst[k].p0 = dy1*widthStep + dx1;
        dst[k].p1 = dy2*widthStep + dx1;
        dst[k].p2 = dy1*widthStep + dx2;
        dst[k].p3 = dy2*widthStep + dx2;
        dst[k].w = src[k][4]/((float)(dx2-dx1)*(dy2-dy1));
    }
}

int
interpolateKeypoint( float N9[3][9], int dx, int dy, int ds, KeyPoint& kpt )
{
    Vec3f b(-(N9[1][5]-N9[1][3])/2,  // Negative 1st deriv with respect to x
            -(N9[1][7]-N9[1][1])/2,  // Negative 1st deriv with respect to y
            -(N9[2][4]-N9[0][4])/2); // Negative 1st deriv with respect to s

    Matx33f A(
        N9[1][3]-2*N9[1][4]+N9[1][5],            // 2nd deriv x, x
        (N9[1][8]-N9[1][6]-N9[1][2]+N9[1][0])/4, // 2nd deriv x, y
        (N9[2][5]-N9[2][3]-N9[0][5]+N9[0][3])/4, // 2nd deriv x, s
        (N9[1][8]-N9[1][6]-N9[1][2]+N9[1][0])/4, // 2nd deriv x, y
        N9[1][1]-2*N9[1][4]+N9[1][7],            // 2nd deriv y, y
        (N9[2][7]-N9[2][1]-N9[0][7]+N9[0][1])/4, // 2nd deriv y, s
        (N9[2][5]-N9[2][3]-N9[0][5]+N9[0][3])/4, // 2nd deriv x, s
        (N9[2][7]-N9[2][1]-N9[0][7]+N9[0][1])/4, // 2nd deriv y, s
        N9[0][4]-2*N9[1][4]+N9[2][4]);           // 2nd deriv s, s

    Vec3f x = A.solve(b, DECOMP_LU);

    bool ok = (x[0] != 0 || x[1] != 0 || x[2] != 0) &&
        std::abs(x[0]) <= 1 && std::abs(x[1]) <= 1 && std::abs(x[2]) <= 1;

    if( ok )
    {
        kpt.pt.x += x[0]*dx;
        kpt.pt.y += x[1]*dy;
        kpt.size = (float)cvRound( kpt.size + x[2]*ds );
    }
    return ok;
}

void findMaximaInLayer( const Mat& sum, const Mat& mask_sum,
                   const vector<Mat>& dets, const vector<Mat>& traces,
                   const vector<int>& sizes, vector<KeyPoint>& keypoints,
                   int octave, int layer, float hessianThreshold, int sampleStep )
{
    // Wavelet Data
    const int NM=1;
    const int dm[NM][5] = { {0, 0, 9, 9, 1} };
    SurfHF Dm;

    int size = sizes[layer];

    // The integral image 'sum' is one pixel bigger than the source image
    int layer_rows = (sum.rows-1)/sampleStep;
    int layer_cols = (sum.cols-1)/sampleStep;

    // Ignore pixels without a 3x3x3 neighbourhood in the layer above
    int margin = (sizes[layer+1]/2)/sampleStep+1;

    if( !mask_sum.empty() )
       resizeHaarPattern( dm, &Dm, NM, 9, size, mask_sum.cols );

    int step = (int)(dets[layer].step/dets[layer].elemSize());

    for( int i = margin; i < layer_rows - margin; i++ )
    {
        const float* det_ptr = dets[layer].ptr<float>(i);
        const float* trace_ptr = traces[layer].ptr<float>(i);
        for( int j = margin; j < layer_cols-margin; j++ )
        {
            float val0 = det_ptr[j];
            if( val0 > hessianThreshold )
            {
                /* Coordinates for the start of the wavelet in the sum image. There
                   is some integer division involved, so don't try to simplify this
                   (cancel out sampleStep) without checking the result is the same */
                int sum_i = sampleStep*(i-(size/2)/sampleStep);
                int sum_j = sampleStep*(j-(size/2)/sampleStep);

                /* The 3x3x3 neighbouring samples around the maxima.
                   The maxima is included at N9[1][4] */

                const float *det1 = &dets[layer-1].at<float>(i, j);
                const float *det2 = &dets[layer].at<float>(i, j);
                const float *det3 = &dets[layer+1].at<float>(i, j);
                float N9[3][9] = { { det1[-step-1], det1[-step], det1[-step+1],
                                     det1[-1]  , det1[0] , det1[1],
                                     det1[step-1] , det1[step] , det1[step+1]  },
                                   { det2[-step-1], det2[-step], det2[-step+1],
                                     det2[-1]  , det2[0] , det2[1],
                                     det2[step-1] , det2[step] , det2[step+1]  },
                                   { det3[-step-1], det3[-step], det3[-step+1],
                                     det3[-1]  , det3[0] , det3[1],
                                     det3[step-1] , det3[step] , det3[step+1]  } };

                /* Check the mask - why not just check the mask at the center of the wavelet? */
                if( !mask_sum.empty() )
                {
                    const int* mask_ptr = &mask_sum.at<int>(sum_i, sum_j);
                    float mval = calcHaarPattern( mask_ptr, &Dm, 1 );
                    if( mval < 0.5 )
                        continue;
                }

                /* Non-maxima suppression. val0 is at N9[1][4]*/
                if( val0 > N9[0][0] && val0 > N9[0][1] && val0 > N9[0][2] &&
                    val0 > N9[0][3] && val0 > N9[0][4] && val0 > N9[0][5] &&
                    val0 > N9[0][6] && val0 > N9[0][7] && val0 > N9[0][8] &&
                    val0 > N9[1][0] && val0 > N9[1][1] && val0 > N9[1][2] &&
                    val0 > N9[1][3]                    && val0 > N9[1][5] &&
                    val0 > N9[1][6] && val0 > N9[1][7] && val0 > N9[1][8] &&
                    val0 > N9[2][0] && val0 > N9[2][1] && val0 > N9[2][2] &&
                    val0 > N9[2][3] && val0 > N9[2][4] && val0 > N9[2][5] &&
                    val0 > N9[2][6] && val0 > N9[2][7] && val0 > N9[2][8] )
                {
                    /* Calculate the wavelet center coordinates for the maxima */
                    float center_i = sum_i + (size-1)*0.5f;
                    float center_j = sum_j + (size-1)*0.5f;

                    KeyPoint kpt( center_j, center_i, (float)sizes[layer],
                                  -1, val0, octave, CV_SIGN(trace_ptr[j]) );

                    /* Interpolate maxima location within the 3x3x3 neighbourhood  */
                    int ds = size - sizes[layer-1];
                    int interp_ok = interpolateKeypoint( N9, sampleStep, sampleStep, ds, kpt );

                    /* Sometimes the interpolation step gives a negative size etc. */
                    if( interp_ok  )
                    {
                        /*printf( "KeyPoint %f %f %d\n", point.pt.x, point.pt.y, point.size );*/
                        //cv::AutoLock lock(findMaximaInLayer_m);
                        keypoints.push_back(kpt);
                    }
                }
            }
        }
    }
}

void calcLayerDetAndTrace( const Mat& sum, int size, int sampleStep,
                                  Mat& det, Mat& trace )
{
    const int NX=3, NY=3, NXY=4;
    const int dx_s[NX][5] = { {0, 2, 3, 7, 1}, {3, 2, 6, 7, -2}, {6, 2, 9, 7, 1} };
    const int dy_s[NY][5] = { {2, 0, 7, 3, 1}, {2, 3, 7, 6, -2}, {2, 6, 7, 9, 1} };
    const int dxy_s[NXY][5] = { {1, 1, 4, 4, 1}, {5, 1, 8, 4, -1}, {1, 5, 4, 8, -1}, {5, 5, 8, 8, 1} };

    SurfHF Dx[NX], Dy[NY], Dxy[NXY];

    if( size > sum.rows-1 || size > sum.cols-1 )
       return;

    resizeHaarPattern( dx_s , Dx , NX , 9, size, sum.cols );
    resizeHaarPattern( dy_s , Dy , NY , 9, size, sum.cols );
    resizeHaarPattern( dxy_s, Dxy, NXY, 9, size, sum.cols );

    /* The integral image 'sum' is one pixel bigger than the source image */
    int samples_i = 1+(sum.rows-1-size)/sampleStep;
    int samples_j = 1+(sum.cols-1-size)/sampleStep;

    /* Ignore pixels where some of the kernel is outside the image */
    int margin = (size/2)/sampleStep;

    for( int i = 0; i < samples_i; i++ )
    {
        const int* sum_ptr = sum.ptr<int>(i*sampleStep);
        float* det_ptr = &det.at<float>(i+margin, margin);
        float* trace_ptr = &trace.at<float>(i+margin, margin);
        for( int j = 0; j < samples_j; j++ )
        {
            float dx  = calcHaarPattern( sum_ptr, Dx , 3 );
            float dy  = calcHaarPattern( sum_ptr, Dy , 3 );
            float dxy = calcHaarPattern( sum_ptr, Dxy, 4 );
            sum_ptr += sampleStep;
            det_ptr[j] = dx*dy - 0.81f*dxy*dxy;
            trace_ptr[j] = dx + dy;
        }
    }
}

struct KeypointGreater
{
    inline bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const
    {
        if(kp1.response > kp2.response) return true;
        if(kp1.response < kp2.response) return false;
        if(kp1.size > kp2.size) return true;
        if(kp1.size < kp2.size) return false;
        if(kp1.octave > kp2.octave) return true;
        if(kp1.octave < kp2.octave) return false;
        if(kp1.pt.y < kp2.pt.y) return false;
        if(kp1.pt.y > kp2.pt.y) return true;
        return kp1.pt.x < kp2.pt.x;
    }
};

struct SURFInvoker : ParallelLoopBody
{
    enum { ORI_RADIUS = 6, ORI_WIN = 60, PATCH_SZ = 20 };

    SURFInvoker( const Mat& _img, const Mat& _sum,
                 vector<KeyPoint>& _keypoints, Mat& _descriptors,
                 bool _extended, bool _upright )
    {
        keypoints = &_keypoints;
        descriptors = &_descriptors;
        img = &_img;
        sum = &_sum;
        extended = _extended;
        upright = _upright;

        // Simple bound for number of grid points in circle of radius ORI_RADIUS
        const int nOriSampleBound = (2*ORI_RADIUS+1)*(2*ORI_RADIUS+1);
		const float SURF_ORI_SIGMA      = 2.5f;
		const float SURF_DESC_SIGMA     = 3.3f;

        // Allocate arrays
        apt.resize(nOriSampleBound);
        aptw.resize(nOriSampleBound);
        DW.resize(PATCH_SZ*PATCH_SZ);

        /* Coordinates and weights of samples used to calculate orientation */
        Mat G_ori = getGaussianKernel( 2*ORI_RADIUS+1, SURF_ORI_SIGMA, CV_32F );
        nOriSamples = 0;
        for( int i = -ORI_RADIUS; i <= ORI_RADIUS; i++ )
        {
            for( int j = -ORI_RADIUS; j <= ORI_RADIUS; j++ )
            {
                if( i*i + j*j <= ORI_RADIUS*ORI_RADIUS )
                {
                    apt[nOriSamples] = cvPoint(i,j);
                    aptw[nOriSamples++] = G_ori.at<float>(i+ORI_RADIUS,0) * G_ori.at<float>(j+ORI_RADIUS,0);
                }
            }
        }
        CV_Assert( nOriSamples <= nOriSampleBound );

        /* Gaussian used to weight descriptor samples */
        Mat G_desc = getGaussianKernel( PATCH_SZ, SURF_DESC_SIGMA, CV_32F );
        for( int i = 0; i < PATCH_SZ; i++ )
        {
            for( int j = 0; j < PATCH_SZ; j++ )
                DW[i*PATCH_SZ+j] = G_desc.at<float>(i,0) * G_desc.at<float>(j,0);
        }
    }

    void operator()(const Range& range) const
    {
        /* X and Y gradient wavelet data */		
		const int   SURF_ORI_SEARCH_INC = 5;
        const int NX=2, NY=2;
        const int dx_s[NX][5] = {{0, 0, 2, 4, -1}, {2, 0, 4, 4, 1}};
        const int dy_s[NY][5] = {{0, 0, 4, 2, 1}, {0, 2, 4, 4, -1}};

        // Optimisation is better using nOriSampleBound than nOriSamples for
        // array lengths.  Maybe because it is a constant known at compile time
        const int nOriSampleBound =(2*ORI_RADIUS+1)*(2*ORI_RADIUS+1);

        float X[nOriSampleBound], Y[nOriSampleBound], angle[nOriSampleBound];
        uchar PATCH[PATCH_SZ+1][PATCH_SZ+1];
        float DX[PATCH_SZ][PATCH_SZ], DY[PATCH_SZ][PATCH_SZ];
        CvMat matX = cvMat(1, nOriSampleBound, CV_32F, X);
        CvMat matY = cvMat(1, nOriSampleBound, CV_32F, Y);
        CvMat _angle = cvMat(1, nOriSampleBound, CV_32F, angle);
        Mat _patch(PATCH_SZ+1, PATCH_SZ+1, CV_8U, PATCH);

        int dsize = extended ? 128 : 64;

        int k, k1 = range.start, k2 = range.end;
        float maxSize = 0;
        for( k = k1; k < k2; k++ )
        {
            maxSize = std::max(maxSize, (*keypoints)[k].size);
        }
        int imaxSize = std::max(cvCeil((PATCH_SZ+1)*maxSize*1.2f/9.0f), 1);
        Ptr<CvMat> winbuf = cvCreateMat( 1, imaxSize*imaxSize, CV_8U );
        for( k = k1; k < k2; k++ )
        {
            int i, j, kk, nangle;
            float* vec;
            SurfHF dx_t[NX], dy_t[NY];
            KeyPoint& kp = (*keypoints)[k];
            float size = kp.size;
            Point2f center = kp.pt;
            /* The sampling intervals and wavelet sized for selecting an orientation
             and building the keypoint descriptor are defined relative to 's' */
            float s = size*1.2f/9.0f;
            /* To find the dominant orientation, the gradients in x and y are
             sampled in a circle of radius 6s using wavelets of size 4s.
             We ensure the gradient wavelet size is even to ensure the
             wavelet pattern is balanced and symmetric around its center */
            int grad_wav_size = 2*cvRound( 2*s );
            if( sum->rows < grad_wav_size || sum->cols < grad_wav_size )
            {
                /* when grad_wav_size is too big,
                 * the sampling of gradient will be meaningless
                 * mark keypoint for deletion. */
                kp.size = -1;
                continue;
            }

            float descriptor_dir = 360.f - 90.f;
            if (upright == 0)
            {
                resizeHaarPattern( dx_s, dx_t, NX, 4, grad_wav_size, sum->cols );
                resizeHaarPattern( dy_s, dy_t, NY, 4, grad_wav_size, sum->cols );
                for( kk = 0, nangle = 0; kk < nOriSamples; kk++ )
                {
                    int x = cvRound( center.x + apt[kk].x*s - (float)(grad_wav_size-1)/2 );
                    int y = cvRound( center.y + apt[kk].y*s - (float)(grad_wav_size-1)/2 );
                    if( y < 0 || y >= sum->rows - grad_wav_size ||
                        x < 0 || x >= sum->cols - grad_wav_size )
                        continue;
                    const int* ptr = &sum->at<int>(y, x);
                    float vx = calcHaarPattern( ptr, dx_t, 2 );
                    float vy = calcHaarPattern( ptr, dy_t, 2 );
                    X[nangle] = vx*aptw[kk];
                    Y[nangle] = vy*aptw[kk];
                    nangle++;
                }
                if( nangle == 0 )
                {
                    // No gradient could be sampled because the keypoint is too
                    // near too one or more of the sides of the image. As we
                    // therefore cannot find a dominant direction, we skip this
                    // keypoint and mark it for later deletion from the sequence.
                    kp.size = -1;
                    continue;
                }
                matX.cols = matY.cols = _angle.cols = nangle;
                cvCartToPolar( &matX, &matY, 0, &_angle, 1 );

                float bestx = 0, besty = 0, descriptor_mod = 0;
                for( i = 0; i < 360; i += SURF_ORI_SEARCH_INC )
                {
                    float sumx = 0, sumy = 0, temp_mod;
                    for( j = 0; j < nangle; j++ )
                    {
                        int d = std::abs(cvRound(angle[j]) - i);
                        if( d < ORI_WIN/2 || d > 360-ORI_WIN/2 )
                        {
                            sumx += X[j];
                            sumy += Y[j];
                        }
                    }
                    temp_mod = sumx*sumx + sumy*sumy;
                    if( temp_mod > descriptor_mod )
                    {
                        descriptor_mod = temp_mod;
                        bestx = sumx;
                        besty = sumy;
                    }
                }
                descriptor_dir = fastAtan2( -besty, bestx );
            }
            kp.angle = descriptor_dir;
            if( !descriptors || !descriptors->data )
                continue;

            /* Extract a window of pixels around the keypoint of size 20s */
            int win_size = (int)((PATCH_SZ+1)*s);
            CV_Assert( winbuf->cols >= win_size*win_size );
            Mat win(win_size, win_size, CV_8U, winbuf->data.ptr);

            if( !upright )
            {
                descriptor_dir *= (float)(CV_PI/180);
                float sin_dir = -std::sin(descriptor_dir);
                float cos_dir =  std::cos(descriptor_dir);

                /* Subpixel interpolation version (slower). Subpixel not required since
                the pixels will all get averaged when we scale down to 20 pixels */
                /*
                float w[] = { cos_dir, sin_dir, center.x,
                -sin_dir, cos_dir , center.y };
                CvMat W = cvMat(2, 3, CV_32F, w);
                cvGetQuadrangleSubPix( img, &win, &W );
                */

                float win_offset = -(float)(win_size-1)/2;
                float start_x = center.x + win_offset*cos_dir + win_offset*sin_dir;
                float start_y = center.y - win_offset*sin_dir + win_offset*cos_dir;
                uchar* WIN = win.data;
#if 0
                // Nearest neighbour version (faster)
                for( i = 0; i < win_size; i++, start_x += sin_dir, start_y += cos_dir )
                {
                    float pixel_x = start_x;
                    float pixel_y = start_y;
                    for( j = 0; j < win_size; j++, pixel_x += cos_dir, pixel_y -= sin_dir )
                    {
                        int x = std::min(std::max(cvRound(pixel_x), 0), img->cols-1);
                        int y = std::min(std::max(cvRound(pixel_y), 0), img->rows-1);
                        WIN[i*win_size + j] = img->at<uchar>(y, x);
                    }
                }
#else
                int ncols1 = img->cols-1, nrows1 = img->rows-1;
                size_t imgstep = img->step;
                for( i = 0; i < win_size; i++, start_x += sin_dir, start_y += cos_dir )
                {
                    double pixel_x = start_x;
                    double pixel_y = start_y;
                    for( j = 0; j < win_size; j++, pixel_x += cos_dir, pixel_y -= sin_dir )
                    {
                        int ix = cvFloor(pixel_x), iy = cvFloor(pixel_y);
                        if( (unsigned)ix < (unsigned)ncols1 &&
                            (unsigned)iy < (unsigned)nrows1 )
                        {
                            float a = (float)(pixel_x - ix), b = (float)(pixel_y - iy);
                            const uchar* imgptr = &img->at<uchar>(iy, ix);
                            WIN[i*win_size + j] = (uchar)
                                cvRound(imgptr[0]*(1.f - a)*(1.f - b) +
                                        imgptr[1]*a*(1.f - b) +
                                        imgptr[imgstep]*(1.f - a)*b +
                                        imgptr[imgstep+1]*a*b);
                        }
                        else
                        {
                            int x = std::min(std::max(cvRound(pixel_x), 0), ncols1);
                            int y = std::min(std::max(cvRound(pixel_y), 0), nrows1);
                            WIN[i*win_size + j] = img->at<uchar>(y, x);
                        }
                    }
                }
#endif
            }
            else
            {
                // extract rect - slightly optimized version of the code above
                // TODO: find faster code, as this is simply an extract rect operation,
                //       e.g. by using cvGetSubRect, problem is the border processing
                // descriptor_dir == 90 grad
                // sin_dir == 1
                // cos_dir == 0

                float win_offset = -(float)(win_size-1)/2;
                int start_x = cvRound(center.x + win_offset);
                int start_y = cvRound(center.y - win_offset);
                uchar* WIN = win.data;
                for( i = 0; i < win_size; i++, start_x++ )
                {
                    int pixel_x = start_x;
                    int pixel_y = start_y;
                    for( j = 0; j < win_size; j++, pixel_y-- )
                    {
                        int x = MAX( pixel_x, 0 );
                        int y = MAX( pixel_y, 0 );
                        x = MIN( x, img->cols-1 );
                        y = MIN( y, img->rows-1 );
                        WIN[i*win_size + j] = img->at<uchar>(y, x);
                    }
                }
            }
            // Scale the window to size PATCH_SZ so each pixel's size is s. This
            // makes calculating the gradients with wavelets of size 2s easy
            resize(win, _patch, _patch.size(), 0, 0, INTER_AREA);

            // Calculate gradients in x and y with wavelets of size 2s
            for( i = 0; i < PATCH_SZ; i++ )
                for( j = 0; j < PATCH_SZ; j++ )
                {
                    float dw = DW[i*PATCH_SZ + j];
                    float vx = (PATCH[i][j+1] - PATCH[i][j] + PATCH[i+1][j+1] - PATCH[i+1][j])*dw;
                    float vy = (PATCH[i+1][j] - PATCH[i][j] + PATCH[i+1][j+1] - PATCH[i][j+1])*dw;
                    DX[i][j] = vx;
                    DY[i][j] = vy;
                }

            // Construct the descriptor
            vec = descriptors->ptr<float>(k);
            for( kk = 0; kk < dsize; kk++ )
                vec[kk] = 0;
            double square_mag = 0;
            if( extended )
            {
                // 128-bin descriptor
                for( i = 0; i < 4; i++ )
                    for( j = 0; j < 4; j++ )
                    {
                        for(int y = i*5; y < i*5+5; y++ )
                        {
                            for(int x = j*5; x < j*5+5; x++ )
                            {
                                float tx = DX[y][x], ty = DY[y][x];
                                if( ty >= 0 )
                                {
                                    vec[0] += tx;
                                    vec[1] += (float)fabs(tx);
                                } else {
                                    vec[2] += tx;
                                    vec[3] += (float)fabs(tx);
                                }
                                if ( tx >= 0 )
                                {
                                    vec[4] += ty;
                                    vec[5] += (float)fabs(ty);
                                } else {
                                    vec[6] += ty;
                                    vec[7] += (float)fabs(ty);
                                }
                            }
                        }
                        for( kk = 0; kk < 8; kk++ )
                            square_mag += vec[kk]*vec[kk];
                        vec += 8;
                    }
            }
            else
            {
                // 64-bin descriptor
                for( i = 0; i < 4; i++ )
                    for( j = 0; j < 4; j++ )
                    {
                        for(int y = i*5; y < i*5+5; y++ )
                        {
                            for(int x = j*5; x < j*5+5; x++ )
                            {
                                float tx = DX[y][x], ty = DY[y][x];
                                vec[0] += tx; vec[1] += ty;
                                vec[2] += (float)fabs(tx); vec[3] += (float)fabs(ty);
                            }
                        }
                        for( kk = 0; kk < 4; kk++ )
                            square_mag += vec[kk]*vec[kk];
                        vec+=4;
                    }
            }

            // unit vector is essential for contrast invariance
            vec = descriptors->ptr<float>(k);
            float scale = (float)(1./(sqrt(square_mag) + DBL_EPSILON));
            for( kk = 0; kk < dsize; kk++ )
                vec[kk] *= scale;
        }
    }

    // Parameters
    const Mat* img;
    const Mat* sum;
    vector<KeyPoint>* keypoints;
    Mat* descriptors;
    bool extended;
    bool upright;

    // Pre-calculated values
    int nOriSamples;
    vector<Point> apt;
    vector<float> aptw;
    vector<float> DW;
};

// Multi-threaded construction of the scale-space pyramid
struct SURFBuildInvoker : ParallelLoopBody
{
    SURFBuildInvoker( const Mat& _sum, const vector<int>& _sizes,
                      const vector<int>& _sampleSteps,
                      vector<Mat>& _dets, vector<Mat>& _traces )
    {
        sum = &_sum;
        sizes = &_sizes;
        sampleSteps = &_sampleSteps;
        dets = &_dets;
        traces = &_traces;
    }

    void operator()(const Range& range) const
    {
        for( int i=range.start; i<range.end; i++ )
            calcLayerDetAndTrace( *sum, (*sizes)[i], (*sampleSteps)[i], (*dets)[i], (*traces)[i] );
    }

    const Mat *sum;
    const vector<int> *sizes;
    const vector<int> *sampleSteps;
    vector<Mat>* dets;
    vector<Mat>* traces;
};

// Multi-threaded search of the scale-space pyramid for keypoints
struct SURFFindInvoker : ParallelLoopBody
{
    SURFFindInvoker( const Mat& _sum, const Mat& _mask_sum,
                     const vector<Mat>& _dets, const vector<Mat>& _traces,
                     const vector<int>& _sizes, const vector<int>& _sampleSteps,
                     const vector<int>& _middleIndices, vector<KeyPoint>& _keypoints,
                     int _nOctaveLayers, float _hessianThreshold )
    {
        sum = &_sum;
        mask_sum = &_mask_sum;
        dets = &_dets;
        traces = &_traces;
        sizes = &_sizes;
        sampleSteps = &_sampleSteps;
        middleIndices = &_middleIndices;
        keypoints = &_keypoints;
        nOctaveLayers = _nOctaveLayers;
        hessianThreshold = _hessianThreshold;
    }
	
    void operator()(const Range& range) const
    {
        for( int i=range.start; i<range.end; i++ )
        {
            int layer = (*middleIndices)[i];
            int octave = i / nOctaveLayers;
            findMaximaInLayer( *sum, *mask_sum, *dets, *traces, *sizes,
                               *keypoints, octave, layer, hessianThreshold,
                               (*sampleSteps)[layer] );
        }
    }

    const Mat *sum;
    const Mat *mask_sum;
    const vector<Mat>* dets;
    const vector<Mat>* traces;
    const vector<int>* sizes;
    const vector<int>* sampleSteps;
    const vector<int>* middleIndices;
    vector<KeyPoint>* keypoints;
    int nOctaveLayers;
    float hessianThreshold;

    static Mutex findMaximaInLayer_m;
};

Mutex SURFFindInvoker::findMaximaInLayer_m;

void fastHessianDetector( const Mat& sum, const Mat& mask_sum, vector<KeyPoint>& keypoints,
                                 int nOctaves, int nOctaveLayers, float hessianThreshold )
{
    /* Sampling step along image x and y axes at first octave. This is doubled
       for each additional octave. WARNING: Increasing this improves speed,
       however keypoint extraction becomes unreliable. */
    const int SAMPLE_STEP0 = 1;
	const int SURF_HAAR_SIZE0 = 9;
	const int SURF_HAAR_SIZE_INC = 6;

    int nTotalLayers = (nOctaveLayers+2)*nOctaves;
    int nMiddleLayers = nOctaveLayers*nOctaves;

    vector<Mat> dets(nTotalLayers);
    vector<Mat> traces(nTotalLayers);
    vector<int> sizes(nTotalLayers);
    vector<int> sampleSteps(nTotalLayers);
    vector<int> middleIndices(nMiddleLayers);

    keypoints.clear();

    // Allocate space and calculate properties of each layer
    int index = 0, middleIndex = 0, step = SAMPLE_STEP0;

    for( int octave = 0; octave < nOctaves; octave++ )
    {
        for( int layer = 0; layer < nOctaveLayers+2; layer++ )
        {
            /* The integral image sum is one pixel bigger than the source image*/
            dets[index].create( (sum.rows-1)/step, (sum.cols-1)/step, CV_32F );
            traces[index].create( (sum.rows-1)/step, (sum.cols-1)/step, CV_32F );
            sizes[index] = (SURF_HAAR_SIZE0 + SURF_HAAR_SIZE_INC*layer) << octave;
            sampleSteps[index] = step;

            if( 0 < layer && layer <= nOctaveLayers )
                middleIndices[middleIndex++] = index;
            index++;
        }
        step *= 2;
    }

    // Calculate hessian determinant and trace samples in each layer
    parallel_for_( Range(0, nTotalLayers),
                   SURFBuildInvoker(sum, sizes, sampleSteps, dets, traces) );

    // Find maxima in the determinant of the hessian
    parallel_for_( Range(0, nMiddleLayers),
                   SURFFindInvoker(sum, mask_sum, dets, traces, sizes,
                                   sampleSteps, middleIndices, keypoints,
                                   nOctaveLayers, hessianThreshold) );

    std::sort(keypoints.begin(), keypoints.end(), KeypointGreater());
}


void mySURF(InputArray _img, InputArray _mask, CV_OUT vector<KeyPoint>& keypoints, OutputArray _descriptors, bool useProvidedKeypoints)
{
	double hessianThreshold = 100;
	int nOctaves = 4;
	int nOctaveLayers = 3;
	bool extended = false;
	bool upright = false;
    Mat img = _img.getMat(), mask = _mask.getMat(), mask1, sum, msum;
    bool doDescriptors = _descriptors.needed();

    CV_Assert(!img.empty() && img.depth() == CV_8U);
    if( img.channels() > 1 )
        cvtColor(img, img, COLOR_BGR2GRAY);

    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.size() == img.size()));
    CV_Assert(hessianThreshold >= 0);
    CV_Assert(nOctaves > 0);
    CV_Assert(nOctaveLayers > 0);

    integral(img, sum, CV_32S);

    // Compute keypoints only if we are not asked for evaluating the descriptors are some given locations:
    if( !useProvidedKeypoints )
    {
        if( !mask.empty() )
        {
            cv::min(mask, 1, mask1);
            integral(mask1, msum, CV_32S);
        }
        fastHessianDetector( sum, msum, keypoints, nOctaves, nOctaveLayers, (float)hessianThreshold );
    }

    int i, j, N = (int)keypoints.size();
    if( N > 0 )
    {
        Mat descriptors;
        bool _1d = false;
        int dcols = extended ? 128 : 64;
        size_t dsize = dcols*sizeof(float);

        if( doDescriptors )
        {
            _1d = _descriptors.kind() == _InputArray::STD_VECTOR && _descriptors.type() == CV_32F;
            if( _1d )
            {
                _descriptors.create(N*dcols, 1, CV_32F);
                descriptors = _descriptors.getMat().reshape(1, N);
            }
            else
            {
                _descriptors.create(N, dcols, CV_32F);
                descriptors = _descriptors.getMat();
            }
        }

        // we call SURFInvoker in any case, even if we do not need descriptors,
        // since it computes orientation of each feature.
        parallel_for_(Range(0, N), SURFInvoker(img, sum, keypoints, descriptors, extended, upright) );

        // remove keypoints that were marked for deletion
        for( i = j = 0; i < N; i++ )
        {
            if( keypoints[i].size > 0 )
            {
                if( i > j )
                {
                    keypoints[j] = keypoints[i];
                    if( doDescriptors )
                        memcpy( descriptors.ptr(j), descriptors.ptr(i), dsize);
                }
                j++;
            }
        }
        if( N > j )
        {
            N = j;
            keypoints.resize(N);
            if( doDescriptors )
            {
                Mat d = descriptors.rowRange(0, N);
                if( _1d )
                    d = d.reshape(1, N*dcols);
                d.copyTo(_descriptors);
            }
        }
    }
}

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
	ifstream fin("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/patternLookup");
    while(!fin.eof()) {
        PatternPoint temp;
        fin >> temp.x >> temp.y >> temp.sigma;
        patternLookup.push_back(temp);
    }
    fin.close();
    fin.open("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/patternSizes");
    int idx = 0;
    while(!fin.eof())
        fin >> patternSizes[idx++];
    fin.close();
    fin.open("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/orientationPairs");
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
    fin.open("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/newDesPairs");
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
	vector<KeyPoint> keypoints;
	ofstream fout;
	/*
	ifstream fin("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/keypoints", ifstream::in);
	while(!fin.eof()) {
		KeyPoint kp;
		fin >> kp.pt.x >> kp.pt.y >> kp.size;
		keypoints.push_back(kp);
	}
	fin.close();
	*/	
	Mat diff_img = imread("D:/project/action/sample_img/diff_img.png", CV_LOAD_IMAGE_GRAYSCALE);
	//BriskFeatureDetector *diff_detector = new BriskFeatureDetector(30); 
	SurfFeatureDetector *diff_detector = new SurfFeatureDetector(30);
	diff_detector->detect(diff_img, keypoints);
	fout.open("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/keypoints_test");
	for(auto y=keypoints.begin(); y!=keypoints.end(); ++y) {
		fout << y->pt.x << " " << y->pt.y << " " << y->size << " " << y->angle << endl;
	}
	fout.close();	
	Mat draw;
	drawKeypoints(diff_img, keypoints, draw, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	imshow("draw_initial", draw);
	//imwrite("D:/project/action/sample_data/draw_initial.png", draw);
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
	fout.open("D:/project/master/MoFREAK_Hardware/mofreak/sample_data/keypoints_final");
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
	*/
	//drawKeypoints(diff_img, keypoints, draw, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	//imshow("draw_after_extraction", draw);
	//imwrite("D:/project/action/sample_data/draw_after_extraction.png", draw);

	cvWaitKey(0);
	//system("pause");
	
	return 0;
}