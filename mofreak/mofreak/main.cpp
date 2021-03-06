#include <string>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <exception>

#include <boost\filesystem.hpp>
#include <boost\thread.hpp>

#include <opencv2\opencv.hpp>

#include "MoFREAKUtilities.h"
#include "Clustering.h"
#include "BagOfWordsRepresentation.h"
#include "SVMInterface.h"

// for debugging the heap (memory leaks, etc)
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

bool DISTRIBUTED = false;

string MOSIFT_DIR, MOFREAK_PATH, VIDEO_PATH, SVM_PATH, METADATA_PATH, RECOG_PATH, RECOG_ONLINE_PATH, TRAINING_PATH, BOW_PATH; // for file structure
string MOFREAK_NEG_PATH, MOFREAK_POS_PATH; // these are TRECVID exclusive
vector<string> labels;
Size newsize(320, 240);
const bool down_sample = false;
bool draw_histogram = true;
bool play_video = true;
struct Interval {
    int start, end;
    string act;
    Interval(int s, int e, string a) : start(s), end(e), act(a) {}
};
vector<Interval> anno;

unsigned int NUM_MOTION_BYTES = 8;
unsigned int NUM_APPEARANCE_BYTES = 8;
unsigned int FEATURE_DIMENSIONALITY = NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES;
unsigned int NUM_CLUSTERS, NUMBER_OF_GROUPS, NUM_CLASSES, ALPHA;

vector<int> possible_classes;
std::deque<MoFREAKFeature> mofreak_ftrs;

enum states {DETECT_MOFREAK, DETECTION_TO_CLASSIFICATION, // standard recognition states
	PICK_CLUSTERS, COMPUTE_BOW_HISTOGRAMS, DETECT, TRAIN, GET_SVM_RESPONSES, RECOGNITION, RECOGNITION_ONLINE, VIDEO_ONLINE, TRAINING, CLASSIFICATION, HISTOGRAM_PARAM, CONVERT_BOW, BOW_TO_SVM}; // these states are exclusive to TRECVID

enum datasets {KTH, TRECVID, HOLLYWOOD, UTI1, UTI2, HMDB51, UCF101, OUR, WEIZMANN};

int dataset = WEIZMANN; //KTH;//HMDB51;
int state = CLASSIFICATION;

MoFREAKUtilities *mofreak;
//SVMInterface svm_interface;
void initialize_label() {
    ifstream fin(SVM_PATH+"/labels.txt");
    if(!fin) cout << "No labels map!!" << endl;
    int clusters, classes, groups;
    fin >> NUM_CLUSTERS >> NUM_CLASSES >> NUMBER_OF_GROUPS;
    string action;
    labels.push_back("Unknown");
    while(fin >> action)
        labels.push_back(action);
    mofreak->set_labels(labels);
    for (unsigned i = 0; i < NUM_CLASSES; ++i)
		possible_classes.push_back(i);
    fin.close();
    cout << "Load labels." << endl;
}
void initialize_anno(const char *video_file) {
    ifstream fin;
    string video_path = path(video_file).generic_string();
    string anno_file = video_path.substr(0, video_path.size()-4)+"_anno.txt";
    fin.open(anno_file);
    if(!fin) cout << "No annotations!!" << endl;
    int s, e;
    string action;
    while(!fin.eof()) {
        fin >> s >> e >> action;
        anno.push_back(Interval(s,e,action));
    }
    fin.close();
    cout << "Load annotations." << endl;
}

string golden(int frame_num) {
    if(anno.empty()) 
        cout << "No annotation!" << endl;
    for(int i=0; i<anno.size(); ++i)
        if(frame_num >= anno[i].start && frame_num <= anno[i].end)
            return anno[i].act;
    return "Unknown";
}

void CVT_RE(Mat &input) {
    cvtColor(input, input, CV_BGR2GRAY);
    if(down_sample) {
        Mat ret(newsize, input.type());
        resize(input, ret, newsize);
        input = ret;
    }
}

struct Detection
{
	int start_frame;
	int end_frame;
	float score;
	string video_name;

	// override the < operator for sorting.
	bool operator < (const Detection &det) const
	{
		return (score < det.score);
	};
};

// set up some hard-coded parameters that are specific to individual datasets.
// these parameters include things like the number of classes and input/output locations.
void setParameters()
{
	if (dataset == TRECVID)
	{
		// event information for TRECVID videos.
		NUM_CLUSTERS = 1000;
		NUM_CLASSES = 2;
		possible_classes.push_back(-1);
		possible_classes.push_back(1);
		ALPHA = 12;

		// structural folder information.
		MOSIFT_DIR = "C:/data/TRECVID/mosift/testing/";
		MOFREAK_PATH = MOSIFT_DIR; // because converting mosift to mofreak just puts them in the same folder as the mosift points. That's fine.
		VIDEO_PATH = "C:/data/TRECVID/mosift/testing/videos/";
		SVM_PATH = "C:/data/TRECVID/svm/";

		// for clustering, separate mofreak into pos and neg examples.
		MOFREAK_NEG_PATH = "C:/data/TRECVID/negative_mofreak_examples/";
		MOFREAK_POS_PATH = "C:/data/TRECVID/positive_mofreak_examples/";
	}

	// KTH
	else if (dataset == KTH)
	{
		NUM_CLUSTERS = 600;
		NUM_CLASSES = 6;
		NUMBER_OF_GROUPS = 25;

		for (unsigned i = 0; i < NUM_CLASSES; ++i)
		{
			possible_classes.push_back(i);
		}

		// structural folder info.
		MOSIFT_DIR = "C:/data/kth/mosift/";
		MOFREAK_PATH = "D:/project/action/dataset/KTH/mofreak"; 
		VIDEO_PATH = "D:/project/action/dataset/KTH/original";
		SVM_PATH = "D:/project/action/dataset/KTH/svm";
		RECOG_PATH = "D:/project/action/dataset/KTH/recognition/";
		RECOG_ONLINE_PATH = "D:/project/action/dataset/KTH/recognition_online/";
		METADATA_PATH = "";
	}
    
    // WEIZMANN
	else if (dataset == WEIZMANN)
	{
		NUM_CLUSTERS = 1000;
		NUM_CLASSES = 10;
		NUMBER_OF_GROUPS = 9;

		for (unsigned i = 0; i < NUM_CLASSES; ++i)
		{
			possible_classes.push_back(i);
		}

		// structural folder info.
		MOSIFT_DIR = "C:/data/kth/mosift/";
		MOFREAK_PATH = "D:/project/action/dataset/Weizmann/mofreak"; 
		VIDEO_PATH = "D:/project/action/dataset/Weizmann/converted";
		SVM_PATH = "D:/project/action/dataset/Weizmann/svm";
		RECOG_PATH = "D:/project/action/dataset/Weizmann/recognition";
		RECOG_ONLINE_PATH = "D:/project/action/dataset/Weizmann/recognition_online";
		METADATA_PATH = "";
	}

	else if (dataset == HMDB51)
	{
		NUM_CLUSTERS = 7000;//5100;
		NUM_CLASSES = 51;//51;

		for (unsigned i = 0; i < NUM_CLASSES; ++i)
		{
			possible_classes.push_back(i);
		}

		// structural folder info
		MOFREAK_PATH = "C:/data/hmdb51/mofreak/";
		VIDEO_PATH = "C:/data/hmdb51/videos/";
		SVM_PATH = "C:/data/hmdb51/svm/";
		METADATA_PATH = "C:/data/hmdb51/metadata/";
	}

	else if (dataset == UCF101)
	{
		NUM_CLUSTERS = 10100;
		NUM_CLASSES = 101;
		NUMBER_OF_GROUPS = 25;

		for (unsigned i = 0; i < NUM_CLASSES; ++i)
		{
			possible_classes.push_back(i);
		}

		MOFREAK_PATH = "D:/project/action/dataset/UCF101/mofreak";
		VIDEO_PATH = "D:/project/action/dataset/UCF101/data";
		SVM_PATH = "D:/project/action/dataset/UCF101/svm";
		BOW_PATH = "D:/project/action/dataset/UCF101/bow";
		METADATA_PATH = "C:/data/ucf101/metadata/";
	}

	else if (dataset == UTI2)
	{
		NUM_CLUSTERS = 600;
		NUM_CLASSES = 6;
		NUMBER_OF_GROUPS = 10;
		for (unsigned i = 0; i < NUM_CLASSES; ++i)
		{
			possible_classes.push_back(i);
		}

		// structural folder info.
		MOFREAK_PATH = "C:/data/UTI/segmented/mofreak/";
		VIDEO_PATH = "C:/data/UTI/segmented/videos/";
		SVM_PATH = "C:/data/UTI/segmented/svm/";
	}
    else {
        MOFREAK_PATH = "D:/project/action/dataset/our3/mofreak";
        SVM_PATH = "D:/project/action/dataset/our3/svm";
        TRAINING_PATH = "D:/project/action/dataset/our3/svm";
        VIDEO_PATH = "D:/project/action/dataset/our3/video";
    }
}

// cluster MoFREAK points to select codewords for a bag-of-words representation.
void cluster()
{
	cout << "Gathering MoFREAK Features..." << endl;
	Clustering clustering(FEATURE_DIMENSIONALITY, NUM_CLUSTERS, 0, NUM_CLASSES, possible_classes, SVM_PATH);
	clustering.setAppearanceDescriptor(NUM_APPEARANCE_BYTES, true);
	clustering.setMotionDescriptor(NUM_MOTION_BYTES, true);

	// for each class
	directory_iterator end_iter;
	for (directory_iterator dir_iter(MOFREAK_PATH); dir_iter != end_iter; ++dir_iter)
	{

		if (is_directory(dir_iter->status()))
		{
			// gather all of the mofreak files.
			string mofreak_action = dir_iter->path().filename().generic_string();
			string action_mofreak_path = MOFREAK_PATH + "/" + mofreak_action;
			mofreak->setCurrentAction(mofreak_action);
			std::cout << "action: " << mofreak_action << std::endl;

			// count the number of mofreak files in this class.
			// that way, we can group them for clustering, to avoid memory issues.
			unsigned int file_count = 0;
			for (directory_iterator file_counter(action_mofreak_path);
				file_counter != end_iter; ++file_counter)
			{
				if (is_regular_file(file_counter->status()))
					file_count++;
			}

			// maximum number of features to read from each file,
			// to avoid reading in too many mofreak features.
			unsigned int features_per_file = 50000/file_count;

			for (directory_iterator mofreak_iter(action_mofreak_path); 
				mofreak_iter != end_iter; ++mofreak_iter)
			{
				// load each mofreak file's data
				if (is_regular_file(mofreak_iter->status()))
				{
					mofreak->readMoFREAKFeatures(mofreak_iter->path().string(), features_per_file);
				}
			}

			// the mofreak features are loaded for this class
			// and now, we select clusters.
			cout << "Building data." << endl;
			clustering.buildDataFromMoFREAK(mofreak->getMoFREAKFeatures(), false, false);
			clustering.randomClusters(true);
			mofreak->clearFeatures();
		}
	}
	clustering.writeClusters(false); // the input argument stands for appending existing file
}

/*
Cluster 1 class at a time, releasing memory in between.
We are having huge memory issues, 
so we can't load it all in at once.
*/
// [DEPRECATED] [TODO]
void clusterHMDB51()
{
	cout << "Gathering MoFREAK Features..." << endl;
	Clustering clustering(FEATURE_DIMENSIONALITY, NUM_CLUSTERS, 0, NUM_CLASSES, possible_classes, SVM_PATH);
	clustering.setAppearanceDescriptor(8, true);
	clustering.setMotionDescriptor(8, true);

	directory_iterator end_iter;

	for (directory_iterator dir_iter(MOFREAK_PATH); dir_iter != end_iter; ++dir_iter)
	{
		// new action.
		if (is_directory(dir_iter->status()))
		{
			std::vector<string> mofreak_files;
			string video_action = dir_iter->path().filename().generic_string();
			cout << "action: " << video_action << endl;

			// set the mofreak object's action to that folder name.
			mofreak->setCurrentAction(video_action);

			// compute mofreak on all files on that folder.
			string action_mofreak_path = MOFREAK_PATH + "/" + video_action;
			cout << "action mofreak path: " << action_mofreak_path << endl;

			for (directory_iterator video_iter(action_mofreak_path); 
				video_iter != end_iter; ++video_iter)
			{
				string filename = video_iter->path().filename().generic_string();
				if (filename.substr(filename.length() - 7, 7) == "mofreak")
				{
					try
					{
						
						mofreak_files.push_back(video_iter->path().string());
						mofreak->readMoFREAKFeatures(mofreak_files.back());
					}
					catch (exception &e)
					{
						cout << "Error: " << e.what() << endl;
						system("PAUSE");
						exit(1);
					}
				}
			}

			// cluster this set, then drop these mofreak features.
			cout << "assign data pts" << endl;
			std::deque<MoFREAKFeature> ftrs = mofreak->getMoFREAKFeatures();
			cout << "got mofreak features" << endl;
			cout << "build data" << endl;
			clustering.buildDataFromMoFREAK(ftrs, false, false);
			cout << "rand clust" << endl;
			clustering.randomClusters(true);
			cout << "clear" << endl;
			mofreak->clearFeatures();
			cout << "cleared" << endl;
		}
	}

	clustering.writeClusters();
}

// Convert a file path (pointing to a mofreak file) into a bag-of-words feature.
void convertFileToBOWFeature(BagOfWordsRepresentation &bow_rep, directory_iterator file_iter)
{
	std::string mofreak_filename = file_iter->path().filename().generic_string();
	if (mofreak_filename.substr(mofreak_filename.length() - 7, 7) == "mofreak")
	{
		bow_rep.convertFileToBOWFeature(file_iter->path().string());
	}
}

void computeBOWRepresentation()
{    
    string labels_path = SVM_PATH + "/labels.txt";
    
    ifstream fin(labels_path);
    if(!fin) {
        vector<string> actions;
        directory_iterator end_iter;
        for (directory_iterator dir_iter(VIDEO_PATH); dir_iter != end_iter; ++dir_iter)
            if(is_directory(dir_iter->status())) 
                actions.push_back(dir_iter->path().filename().generic_string());
        ofstream fout(labels_path);
        fout << actions.size()*100 << " " << actions.size() << " " << "1";
        for(auto it=actions.begin(); it!=actions.end(); ++it)
            fout << endl << *it;
        fout.close();  
    }
    fin.close();
    
	// initialize BOW representation
	BagOfWordsRepresentation bow_rep(NUM_CLUSTERS, NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES, SVM_PATH, NUMBER_OF_GROUPS, dataset);
	bow_rep.intializeBOWMemory(SVM_PATH);

	// load mofreak files
	std::cout << "Gathering MoFREAK files from " << MOFREAK_PATH << std::endl;
	std::vector<std::string> mofreak_files;
	directory_iterator end_iter; 

#pragma omp parallel 
	{
		for (directory_iterator dir_iter(MOFREAK_PATH); dir_iter != end_iter; ++dir_iter)
		{
			// if organized by directories, process the entire subdirectory.
			if (is_directory(dir_iter->status()))
			{
				std::string action = dir_iter->path().filename().generic_string();
				std::string action_mofreak_path = MOFREAK_PATH + "/" + action;
				std::cout << "action: " << action << std::endl;

				for (directory_iterator mofreak_iter(action_mofreak_path); mofreak_iter != end_iter; ++mofreak_iter)
				{
					if (is_regular_file(mofreak_iter->status()))
					{
#pragma omp single nowait
						{
							convertFileToBOWFeature(bow_rep, mofreak_iter);
						}
					}
				}
			}

			// otherwise, if all of the mofreak files are in one large directory,
			// process each individual file independently.
			else if (is_regular_file(dir_iter->status()))
			{
#pragma omp single nowait
				{
					convertFileToBOWFeature(bow_rep, dir_iter);
				}
			}
		}
	}


	/*
	We've looped over all the MoFREAK files and generated the BOW features,
	along with the cross-validation groupings.
	To finish off, we simply stream these groupings out to files.
	The printing function cleans up the open files, as well.
	*/
	bow_rep.writeBOWFeaturesToFiles();
	std::cout << "Completed printing bag-of-words representation to files" << std::endl;
}

double classify()
{
	cout << "in eval" << endl;
	// gather testing and training files...
	vector<std::string> testing_files;
	vector<std::string> training_files;
	cout << "Eval SVM..." << endl;
	directory_iterator end_iter;

	for (directory_iterator dir_iter(SVM_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 5, 5) == "train")
			{
				training_files.push_back(current_file.string());
			}
			else if (filename.substr(filename.length() - 4, 4) == "test")
			{
				testing_files.push_back(current_file.string());
			}
		}
	}

	// evaluate the SVM with leave-one-out.
    string results_file = SVM_PATH;
	results_file.append("/svm_results.txt");
	ofstream output_file(results_file);
    char name[1024];
    
    vector<string> model_file_name(training_files.size(), SVM_PATH);
    for(int i=0; i<model_file_name.size(); ++i) {
        sprintf(name, "/model_%d.svm", i+1);
        model_file_name[i] += name;
    }
    
    vector<string> svm_out(training_files.size(), SVM_PATH);
    for(int i=0; i<svm_out.size(); ++i) {
        sprintf(name, "/response_%d.txt", i+1);
        svm_out[i] += name;
    }

	// confusion matrix.
	cv::Mat confusion_matrix = cv::Mat::zeros(NUM_CLASSES, NUM_CLASSES, CV_32F);

	double summed_accuracy = 0.0;
	for (unsigned i = 0; i < training_files.size(); ++i)
	{
		cout << "New loop iteration" << endl;
		SVMInterface svm_guy;
		// tell GUI where we're at in the l-o-o process
		cout << "Cross validation set " << i + 1 << endl;

		// build model.
		string training_file = training_files[i];
		svm_guy.trainModel(training_file, model_file_name[i]);

		// get accuracy.
		string test_filename = testing_files[i];
		double accuracy = svm_guy.testModel(test_filename, model_file_name[i], svm_out[i]);
		summed_accuracy += accuracy;

		// update confusion matrix.
		// get svm responses.
		vector<int> responses;

		ifstream response_file(svm_out[i]);
		string line;
		while (std::getline(response_file, line))
		{
			int response;
			istringstream(line) >> response;
			responses.push_back(response);
		}
		response_file.close();

		// now get expected output.
		vector<int> ground_truth;

		ifstream truth_file(test_filename);
		while (std::getline(truth_file, line))
		{
			int truth;
			int first_space = line.find(" ");
			if (first_space != string::npos)
			{
				istringstream (line.substr(0, first_space)) >> truth;
				ground_truth.push_back(truth);
			}
		}

		// now add that info to the confusion matrix.
		// row = ground truth, col = predicted..
		for (unsigned int response = 0; response < responses.size(); ++response)
		{
			int row = ground_truth[response] - 1;
			int col = responses[response] - 1;

			confusion_matrix.at<float>(row, col) += 1;
		}
		
		// debugging...print to testing file.
		output_file << training_files[i] <<", " << testing_files[i] << ", " << accuracy << std::endl;
	}	

	// normalize each row.
	// NUM_CLASSES rows/cols (1 per action)
	for (unsigned int row = 0; row < NUM_CLASSES; ++row)
	{
		float normalizer = 0.0;
		for (unsigned int col = 0; col < NUM_CLASSES; ++col)
		{
			normalizer += confusion_matrix.at<float>(row, col);
		}

		for (unsigned int col = 0; col < NUM_CLASSES; ++col)
		{
			confusion_matrix.at<float>(row, col) /= normalizer;
		}
	}

	cout << "Confusion matrix" << endl << "---------------------" << endl;
	for (int row = 0; row < confusion_matrix.rows; ++row)
	{
		for (int col = 0; col < confusion_matrix.cols; ++col)
		{
			cout << confusion_matrix.at<float>(row, col) << ", ";
		}
		cout << endl << endl;
	}

	output_file.close();

	// output average accuracy.
	double denominator = (double)training_files.size();
	double average_accuracy = summed_accuracy/denominator;

	cout << "Averaged accuracy: " << average_accuracy << "%" << endl;

	/*
	write accuracy to file.  
	temporary for testing.
	*/

	ofstream acc_file;
	acc_file.open("accuracy.txt");
	
	acc_file << average_accuracy;
	acc_file.close();

	return average_accuracy;
}

// exclusively used for the TRECVID scenario now.
// otherwise, [DEPRECATED][TODO]
void pickClusters()
{
	// load all MoFREAK files.
	// So, we will have one folder with all MoFREAK files in it.  Simple...
	cout << "Gathering MoFREAK Features..." << endl;
	vector<std::string> mofreak_files;

	// POSITIVE EXAMPLES
	directory_iterator end_iter;

	for (directory_iterator dir_iter(MOFREAK_POS_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 7, 7) == "mofreak")
			{
				mofreak->readMoFREAKFeatures(current_file.string());
			}
		}
	}

	mofreak->setAllFeaturesToLabel(1);
	mofreak_ftrs = mofreak->getMoFREAKFeatures();

	// NEGATIVE EXAMPLES
	MoFREAKUtilities negative_mofreak(dataset);

	for (directory_iterator dir_iter(MOFREAK_NEG_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 7, 7) == "mofreak")
			{
				negative_mofreak.readMoFREAKFeatures(current_file.string());
			}
		}
	}

	negative_mofreak.setAllFeaturesToLabel(-1);
	std::deque<MoFREAKFeature> negative_ftrs = negative_mofreak.getMoFREAKFeatures();

	// append the negative features to the end of the positive ones.
	mofreak_ftrs.insert(mofreak_ftrs.end(), negative_ftrs.begin(), negative_ftrs.end());
	cout << "MoFREAK features gathered." << endl;

	// Do random cluster selection.
	cv::Mat data_pts(mofreak_ftrs.size(), FEATURE_DIMENSIONALITY, CV_32FC1);

	Clustering clustering(FEATURE_DIMENSIONALITY, NUM_CLUSTERS, 1, NUM_CLASSES, possible_classes, SVM_PATH);
	clustering.setAppearanceDescriptor(8, true);
	clustering.setMotionDescriptor(8, true);

	cout << "Formatting features..." << endl;
	clustering.buildDataFromMoFREAK(mofreak_ftrs, false, false, false);

	cout << "Clustering..." << endl;
	clustering.randomClusters();

	// print clusters to file
	cout << "Writing clusters to file..." << endl;
	clustering.writeClusters();
	cout << "Clusters written." << endl;

	data_pts.release();
}

/* for this, organize mofreak files into pos + neg folders and do them separately.
// use openmp to parallelize each file's BOW stuff.
// give each one it's own libsvm file to output ot, so we don't get any conflicts.
// we will merge at the end with python.

// so, this function will give us sliding window BOW features.
// We can also use this to get our SVM responses to mean-shift away.
// ***********
// Exclusively used for the TRECVID scenario now,
 any remaining examples are deprecated. [TODO]*/
void computeBOWHistograms(bool positive_examples)
{
	// gather all files int vector<string> mofreak_files
	cout << "Gathering MoFREAK Features..." << endl;
	vector<std::string> mofreak_files;
	directory_iterator end_iter;

	if (DISTRIBUTED)
	{
		MOFREAK_PATH = "mosift/";
	}

	for (directory_iterator dir_iter(MOFREAK_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 7, 7) == "mofreak")
			{
				mofreak_files.push_back(current_file.string());
			}
		}
	}
	cout << "MoFREAK features gathered." << endl;

	// load clusters.
	BagOfWordsRepresentation bow_rep(NUM_CLUSTERS, FEATURE_DIMENSIONALITY, SVM_PATH, NUMBER_OF_GROUPS, dataset);

	// for each file....
	// slide window of length alpha and use those pts to create a BOW feature.
#pragma omp parallel for
	for (int i = 0; i < mofreak_files.size(); ++i)
	{
		cout << "Computing on " << mofreak_files[i] << endl;
		std::string bow_file = mofreak_files[i];
		bow_file.append(".test");
		ofstream bow_out(bow_file);

		int label = positive_examples ? 1 : -1;
		bow_rep.computeSlidingBagOfWords(mofreak_files[i], ALPHA, label, bow_out);
		bow_out.close();
		cout << "Done " << mofreak_files[i] << endl;
	}
}

void detectEvents()
{
	vector<std::string> response_files;
	directory_iterator end_iter;

	for (directory_iterator dir_iter(SVM_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.length() > 9)
			{
				if (filename.substr(filename.length() - 13, 13) == "responses.txt")
				{
					response_files.push_back(current_file.string());
				}
			}
		}
	}

	for (auto it = response_files.begin(); it != response_files.end(); ++it)
	{
		cout << "filename: " << *it << endl;
		// read in libsvm output.
		ifstream svm_in(*it);

		// store each value in a list that we can reference.
		vector<float> svm_responses;
		while (!svm_in.eof())
		{
			float response;
			svm_in >> response;
			svm_responses.push_back(response);
		}

		cout << svm_responses.size() << " total SVM responses." << endl;


		// get peaks. [val(x) > val(x - 1) & val(x) > val(x + 1)]
		vector<int> peak_indices;
		for (unsigned int i = 1; i < svm_responses.size() - 1; ++i)
		{
			float response_x = svm_responses[i];
			float response_x_minus_1 = svm_responses[i - 1];
			float response_x_plus_1 = svm_responses[i + 1];

			if ((response_x > response_x_minus_1) && (response_x > response_x_plus_1))
			{
				peak_indices.push_back(i);
			}
		}

		cout << peak_indices.size() << " total detected peaks" << endl;

		// For each of those peaks, run the meanshift-like process to determine if its a window-wise local maxima in the response space.
		// that is, check the alpha/2 points before it and alpha/2 points after it.  If it is the largest value in that window,
		// then this is a candidate detection.
		vector<int> candidate_indices;
		for (auto peak = peak_indices.begin(); peak != peak_indices.end(); ++peak)
		{
			double value = svm_responses[*peak];
			int start_index = max((*peak) - (int)ALPHA, 0);
			int end_index = min((*peak) + (int)ALPHA, (int)svm_responses.size() - 1);
			bool is_local_max = true;

			for (int i = start_index; i < end_index; ++i)
			{
				if (svm_responses[i] > value)
				{
					is_local_max = false;
					break;
				}
			}

			if (is_local_max)
			{
				candidate_indices.push_back(*peak);
			}
		}

		cout << candidate_indices.size() << " detected candidates" << endl;


		// finally, if the detection's response is above our defined threshold, it's a real detection.
		float THRESHOLD = 0;
		unsigned int MAX_DETECTIONS = 30;
		unsigned int MIN_DETECTIONS = 1;
		float STEP = 0.05;
		bool PREVIOUSLY_LOWERED = true;
		bool FIRST_TRY = true;
		// trying an optimization metric for the THRESHOLD.  Say...we want 50 detections per video,
		// we will optimize until that's right.
		
		vector<Detection> detections;
		while (true)
		{
			for (auto candidate = candidate_indices.begin(); candidate != candidate_indices.end(); ++candidate)
			{
				if (svm_responses[*candidate] > THRESHOLD)
				{
					// the BOW feature stretches from the root point (*it) to alpha away.  So if alpha is 10 and it's the first response,
					// it would be keyframes 0 to 10 (or frames 0 to 50).
					int end_index = (*candidate) + ALPHA;
				
					Detection detection;
					detection.start_frame = (*candidate) * 5;
					detection.end_frame = end_index * 5;
					detection.score = svm_responses[*candidate];
					detection.video_name = "GenericVideoName.mpg"; // [TODO]

					detections.push_back(detection);
				}
			}

			unsigned int num_detections = detections.size();
			cout << num_detections << " detections" << endl;

			if (num_detections < MIN_DETECTIONS)
			{
				// maybe there aren't enough candidates.
				if (candidate_indices.size() < MIN_DETECTIONS)
				{
					break;
					//MIN_DETECTIONS = 3 * candidate_indices.size()/4;
				}
				// too few detections, lower the threshold to allow for more.
				if (FIRST_TRY || PREVIOUSLY_LOWERED)
				{
					THRESHOLD -= STEP;
					PREVIOUSLY_LOWERED = true;
					FIRST_TRY = false;
				}
				else
				{
					// we raised it last time to allow less, but not not enough.
					// shrink the step size to get a finer grain.
					STEP -= 0.005;
					THRESHOLD -= STEP;
					PREVIOUSLY_LOWERED = true;
				}
				cout << "STEP: " << STEP << ", THRESHOLD: " << THRESHOLD << endl;
				detections.clear();
			}
			else if (num_detections > MAX_DETECTIONS)
			{
				// too many detections, raise threshold to allow less.
				if (FIRST_TRY || !PREVIOUSLY_LOWERED)
				{
					THRESHOLD += STEP;
					FIRST_TRY = false;
					PREVIOUSLY_LOWERED = false;
				}
				else
				{
					// we lowered it last time to allow more, but now we have too many.
					// shrink the step size grain for finer detail and raise the threshold by this new amount.
					STEP += 0.005;
					THRESHOLD += STEP;
					PREVIOUSLY_LOWERED = false;
				}
				detections.clear();
				cout << "STEP:" << STEP << ", THRESHOLD: " << THRESHOLD << endl;
			}
			else
			{
				// we are in the desired detection range.
				// now we can sort and print them.
				cout << "Accepting a threshold of " << THRESHOLD << " that permits " << num_detections << " events." << endl;
				break;
			}
		}
		// sort by likelihood
		std::sort(detections.begin(), detections.end());
		std::reverse(detections.begin(), detections.end());

		// print to file
		ofstream detection_stream(*it + ".detections");
		for (auto det = detections.begin(); det != detections.end(); ++det)
		{
			detection_stream << *it << ", " << det->start_frame << ", " << det->end_frame << ", " << det->score << endl;
		}
		detection_stream.close();
		cout << "-----------------------------------" << endl << endl;
	}
}

void trainTRECVID()
{
	SVMInterface svm;
	string model_file_name = "C:/data/TRECVID/svm/model.svm";
	svm.trainModel(SVM_PATH + "/training.train", model_file_name);
}

// For TRECVID detections
void computeSVMResponses()
{
	SVMInterface svm;
	directory_iterator end_iter;
	string model_file = SVM_PATH + "/model.svm";

	if (DISTRIBUTED)
	{
		SVM_PATH = "mosift/";
		model_file = "C:/data/model.svm";
	}

	cout << "SVM_PATH: " << SVM_PATH << endl;

	for (directory_iterator dir_iter(SVM_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 4, 4) == "test")
			{
				string test_file = SVM_PATH + "/" + filename;
				cout << "Testing SVM with file " << test_file << " with model " << model_file << endl;
				svm.testModelTRECVID(test_file,  model_file);
			}
		}
	}
}

// given a collection of videos, generate a single mofreak file per video,
// containing the descriptor data for that video.
void computeMoFREAKFiles()
{
	directory_iterator end_iter;

	cout << "Here are the videos: " << VIDEO_PATH << endl;
	cout << "MoFREAK files will go here: " << MOFREAK_PATH << endl;
	cout << "Motion bytes: " << NUM_MOTION_BYTES << endl;
	cout << "Appearance bytes: " << NUM_APPEARANCE_BYTES << endl;
	for (directory_iterator dir_iter(VIDEO_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			// parse mosift files so first x characters gets us the video name.
			path current_file = dir_iter->path();
			string video_path = current_file.generic_string();
			string video_filename = current_file.filename().generic_string();

			if ((video_filename.substr(video_filename.length() - 3, 3) == "avi"))
			{
				cout << "AVI: " << VIDEO_PATH << "/" << video_filename << endl;

				string video = VIDEO_PATH + "/" + video_filename;
				string mofreak_path = MOFREAK_PATH + "/" + video_filename + ".mofreak";

				mofreak->computeMoFREAKFromFile(video, mofreak_path, true);
			}
		}
		else if (is_directory(dir_iter->status()))
		{
			// get folder name.
			string video_action = dir_iter->path().filename().generic_string();
			cout << "action: " << video_action << endl;

			// set the mofreak object's action to that folder name.
			mofreak->setCurrentAction(video_action);

			// compute mofreak on all files on that folder.
			string action_video_path = VIDEO_PATH + "/" + video_action;
			//cout << "action video path: " << action_video_path << endl;

			for (directory_iterator video_iter(action_video_path); 
				video_iter != end_iter; ++video_iter)
			{
				if (is_regular_file(video_iter->status()))
				{
					string video_filename = video_iter->path().filename().generic_string();
					if (video_filename.substr(video_filename.length() - 3, 3) == "avi")
					{
						//cout << "filename: " << video_filename << endl;
						//cout << "AVI: " << action_video_path << video_filename << endl;

						string mofreak_path = MOFREAK_PATH + "/" + video_action + "/" + video_filename + ".mofreak";

						// create the corresponding directories, then go ahead and compute the mofreak files.
						boost::filesystem::path dir_to_create(MOFREAK_PATH + "/" + video_action + "/");
						boost::system::error_code returned_error;
						boost::filesystem::create_directories(dir_to_create, returned_error);
						if (returned_error)
						{
							std::cout << "Could not make directory " << dir_to_create.string() << std::endl;
							exit(1);
						}

						//cout << "mofreak path: " << mofreak_path << endl;
						mofreak->computeMoFREAKFromFile(action_video_path + "/" + video_filename, mofreak_path, true);
					}
				}
			}
		}
	}
}

void recognition(const char *video_file) {
    clock_t start = clock();
    clock_t time_mofreak, time_BOW, time_predict;
    
    // Initialize file path
    //SVM_PATH = TRAINING_PATH;
    //SVM_PATH = "D:/project/action/dataset/KTH/saved/thesis/typical_BRISK30_85/svm/";
    initialize_label();
    string video_filename = path(video_file).filename().generic_string();
    string mofreak_path = RECOG_PATH + "/" + video_filename + ".mofreak";
    BagOfWordsRepresentation bow_rep(NUM_CLUSTERS, NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES, SVM_PATH, NUMBER_OF_GROUPS, dataset);    
    SVMInterface svm_guy;
    //string model_path = SVM_PATH + "/model_18.svm";
    string model_path = SVM_PATH + "/1.train.model";
    //string svm_out = RECOG_PATH + "/response.txt";
    
    // Compute MoFREAK feature from file
    start = clock();
    mofreak->computeMoFREAKFromFile(video_file, mofreak_path, true);
    time_mofreak = clock()-start;
    
    // Compute BOW from MoFREAK and save .bow file
    // clusters.txt is specified in SVM_PATH/clusters.txt by initialization of bow_rep
	bool success;
    Mat bow_feature;
    start = clock();
	try
	{
		bow_feature = bow_rep.buildHistogram(mofreak_path, success);
	}
	catch (cv::Exception &e)
	{
		cout << "Error: " << e.what() << endl;
		exit(1);
	}
	if (!success)
	{
		std::cout << "Bag-of-words feature construction was unsuccessful.  Investigate." << std::endl;
		exit(1);
	}      
    time_BOW = clock() - start;
    
    // Using trained SVM model to predict label of input video
    svm_node *x = new svm_node[bow_feature.cols+1];    
	for (int col = 0; col < bow_feature.cols; ++col)
	{
        x[col].index = col+1;
        x[col].value = (double)bow_feature.at<float>(0, col);
	}
    x[bow_feature.cols].index = -1;    
    svm_model *model = svm_load_model(model_path.c_str());
    start = clock();
    double predict_label = svm_predict(model, x);
    time_predict = clock() - start;
    delete [] x;
    cout << "label: " << labels[predict_label] << endl;
    cout << "Build MoFREAK duration: " << time_mofreak/(double)CLOCKS_PER_SEC << " seconds" << endl;
    cout << "Compute BOW duration: " << time_BOW/(double)CLOCKS_PER_SEC << " seconds" << endl;
    cout << "SVM predict duration: " << time_predict/(double)CLOCKS_PER_SEC << " seconds" << endl;
    
    stringstream ss;
	ss << (0) << " "; // label for svm
	for (int col = 0; col < bow_feature.cols; ++col)
	{
		ss << (int)(col + 1) << ":" << (float)bow_feature.at<float>(0, col) << " ";
	}
	string current_line;
	current_line = ss.str();
	ss.str("");
	ss.clear();
    
    ofstream fout;
    string recognition_path = RECOG_PATH + "/" + video_filename.substr(0, video_filename.length() - 4) + ".bow";
    fout.open(recognition_path);
    fout << current_line << endl;
    fout.close();
    // Play video
    /*
    VideoCapture video;
    video.open(video_file);
    if(!video.isOpened()) {
        cout << "Fail to open " << video_file << "!" << endl;
        return;
    }
    while(true) {
        Mat frame;
        if(!video.read(frame)) break;
        imshow("Test Video", frame);
        waitKey(10);
    }
    destroyWindow("Test Video");
    */
    //waitKey(0);
}

void recognition_online(const char *video_file, const int delta_f, const int delta_h) {
    const int GAP_FOR_FRAME_DIFFERENCE = 5;    
    int key;
    // Initialize file path
    //SVM_PATH = TRAINING_PATH;
    //string model_path = SVM_PATH + "/model.svm";
    SVM_PATH = "D:/project/action/dataset/KTH/saved/thesis/typical_BRISK30_85/svm";
    string model_path = SVM_PATH + "/model_18.svm";
    initialize_label();
    initialize_anno(video_file);
    string video_filename = path(video_file).filename().generic_string();
    string mofreak_path = RECOG_PATH + "/" + video_filename + ".mofreak";
    BagOfWordsRepresentation bow_rep(NUM_CLUSTERS, NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES, SVM_PATH, NUMBER_OF_GROUPS, dataset);    
    SVMInterface svm_guy;
    ofstream fout;
    
    VideoCapture capture;
    capture.open(video_file);
	if (!capture.isOpened())
		cout << "Could not open file: " << video_filename << endl;
    
    Mat current_frame;
    Mat prev_frame;
    Mat original_frame;
    queue<Mat> frame_queue;
    int queue_num = 1;
	for (unsigned int i = 0; i < GAP_FOR_FRAME_DIFFERENCE; ++i)
	{
		capture >> prev_frame; // ignore first 'GAP_FOR_FRAME_DIFFERENCE' frames.  Read them in and carry on.
        CVT_RE(prev_frame);
		frame_queue.push(prev_frame.clone());
	}
	prev_frame = frame_queue.front();
	frame_queue.pop();
    unsigned int frame_num = GAP_FOR_FRAME_DIFFERENCE - 1;    
    
    queue<Mat> histogram_queue;
    Mat curr_histogram = Mat::zeros(1, NUM_CLUSTERS, CV_32FC1);
    Mat total_histogram = Mat::zeros(1, NUM_CLUSTERS, CV_32FC1);
    int total_N = 0;
    Mat curr_bow = Mat::zeros(1, NUM_CLUSTERS, CV_32FC1);
    svm_model *model = svm_load_model(model_path.c_str());
    svm_node *x = new svm_node[curr_bow.cols+1];  
    double predict_label = 0.0;
    int fps = capture.get(CV_CAP_PROP_FPS);
    int action, person, video_number;
    int best_match;
    clock_t start_frame, time_frame;
    clock_t start_predict, time_predict;
    int total_kps = 0;
    int total_accu = 0, accu = 0;
    string true_act = "Unknown";
    
	BRISK *diff_detector = new BRISK(30); 
    //cv::SurfFeatureDetector *diff_detector = new cv::SurfFeatureDetector(30);    
    start_predict = clock();
    while(true) {
        start_frame = clock();
        capture >> current_frame;
        if (current_frame.empty())
            break;
		++frame_num;
        original_frame = current_frame;
        CVT_RE(current_frame);
        Mat diff_img(current_frame.rows, current_frame.cols, CV_8U);
        absdiff(current_frame, prev_frame, diff_img);
        vector<KeyPoint> keypoints, diff_keypoints;
        Mat descriptors;
        diff_detector->detect(diff_img, keypoints);
        for(auto keypt = keypoints.begin(); keypt != keypoints.end();) {
            if(!mofreak->sufficientMotion(current_frame, prev_frame, keypt->pt.x, keypt->pt.y, keypt->size))
                keypt=keypoints.erase(keypt);
            else
                ++keypt;
        }        
        mofreak->myFREAKcompute(diff_img, keypoints, descriptors);
		unsigned char *pointer_to_descriptor_row = 0;
		unsigned int keypoint_row = 0;
        total_kps += keypoints.size();
		for (auto keypt = keypoints.begin(); keypt != keypoints.end(); ++keypt)
		{
			pointer_to_descriptor_row = descriptors.ptr<unsigned char>(keypoint_row);
			int motion = 0;
            MoFREAKFeature ftr(mofreak->NUMBER_OF_BYTES_FOR_MOTION, mofreak->NUMBER_OF_BYTES_FOR_APPEARANCE);
            ftr.frame_number = frame_num;
            ftr.scale = keypt->size;
            ftr.x = keypt->pt.x;
            ftr.y = keypt->pt.y;
            Mat feature_vector(1, FEATURE_DIMENSIONALITY, CV_8U);

            for (unsigned i = 0; i < mofreak->NUMBER_OF_BYTES_FOR_APPEARANCE; ++i)
            {
                ftr.appearance[i] = pointer_to_descriptor_row[i];
                feature_vector.at<unsigned char>(0, i) = pointer_to_descriptor_row[i];
            }

            vector<unsigned int> motion_desc;
            
            mofreak->extractMotionByMotionInterchangePatterns(current_frame, prev_frame, motion_desc, keypt->size, keypt->pt.x, keypt->pt.y);

            for (unsigned i = 0; i < mofreak->NUMBER_OF_BYTES_FOR_MOTION; ++i)
            {
                ftr.motion[i] = motion_desc[i];
                feature_vector.at<unsigned char>(0, mofreak->NUMBER_OF_BYTES_FOR_APPEARANCE+i) = motion_desc[i];
            }

            mofreak->readMetadata(video_filename, action, video_number, person);

            ftr.action = action;
            ftr.video_number = video_number;
            ftr.person = person;

            ftr.motion_x = 0;
            ftr.motion_y = 0;

            mofreak->features.push_back(ftr);
			keypoint_row++;
            
            best_match = bow_rep.bruteForceMatch(feature_vector);
            curr_histogram.at<float>(0, best_match) += 1;
            total_histogram.at<float>(0, best_match) += 1;
            total_N += 1;
        }      
		frame_queue.push(current_frame.clone());
		prev_frame = frame_queue.front();
		frame_queue.pop();
        histogram_queue.push(curr_histogram.clone());
        curr_histogram = Mat::zeros(1, NUM_CLUSTERS, CV_32FC1);
        if(histogram_queue.size() > delta_h) {
            Mat temp = histogram_queue.front();
            histogram_queue.pop();
            for(int i=0; i<NUM_CLUSTERS; ++i) {
                total_histogram.at<float>(0, i) -= temp.at<float>(0, i);
                total_N -= (int)temp.at<float>(0, i);
            }
        }
        ++queue_num;
        if(queue_num > delta_f) {
            queue_num = 1;
            for(int i=0; i<NUM_CLUSTERS; ++i)
                curr_bow.at<float>(0, i) = (float)total_histogram.at<float>(0, i) / total_N;
            // Compute BOW from MoFREAK and save .bow file
            // clusters.txt is specified in SVM_PATH/clusters.txt by initialization of bow_rep                                 
            for (int col = 0; col < curr_bow.cols; ++col)
            {
                x[col].index = col+1;
                x[col].value = (double)curr_bow.at<float>(0, col);
            }
            x[curr_bow.cols].index = -1;    
            predict_label = svm_predict(model, x);
            time_predict = clock()-start_predict;
            true_act = golden(frame_num);
            if(keypoints.size() < 20) {
            //if(keypoints.empty()) { //for KTH
                predict_label = 0; 
                true_act = "Unknown";
            }
            cout << "Frame Number: " << frame_num << " Label: " << labels[predict_label] << " Golden: " << true_act << "   Time: " << (double)time_predict/CLOCKS_PER_SEC << endl;
            start_predict = clock();
            total_kps = 0;
            ++total_accu;
            if(labels[predict_label] == true_act) ++accu;
            
            if(draw_histogram) {
                Mat hisImage = Mat::ones(256, NUM_CLUSTERS, CV_8U)*255;
                normalize(total_histogram, curr_bow, 0, hisImage.rows, NORM_MINMAX);
                
                for( int i = 0; i < NUM_CLUSTERS; i++ )
                    rectangle( hisImage, Point(i, hisImage.rows), Point((i+1), hisImage.rows - cvRound(curr_bow.at<float>(i))), Scalar::all(0), -1, 8, 0 );
                imshow("histogram", hisImage);
            }
        }
        if(play_video) {
            if(labels[predict_label] == true_act)
                putText(original_frame, labels[predict_label], cvPoint(10,30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,0), 2);
            else
                putText(original_frame, labels[predict_label], cvPoint(10,30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 2);
            imshow("Test Video", original_frame);
            time_frame = clock()-start_frame;
            if(1000/fps-time_frame > 0)
                key = waitKey(1000/fps-time_frame);
            else
                key = waitKey(1);
            if(key==27) break;
        }
	} 
    delete [] x;
    delete diff_detector;
    cout << "Accuracy: " << (float)accu/total_accu << endl;
}

void video_online() {
    const int GAP_FOR_FRAME_DIFFERENCE = 5;
    const int delta_h = 60;
    const int delta_f = 10;
    int key;
    
    // Initialize file path
    //SVM_PATH = "D:/project/action/dataset/KTH/saved/thesis/typical_BRISK30_85/svm/";
    //string model_path = SVM_PATH + "/model_18.svm";  
    SVM_PATH = TRAINING_PATH;
    string model_path = SVM_PATH + "/model.svm";  
    initialize_label();
    //string video_filename = path(video_file).filename().generic_string();
    //string mofreak_path = RECOG_PATH + "/" + video_filename + ".mofreak";
    BagOfWordsRepresentation bow_rep(NUM_CLUSTERS, NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES, SVM_PATH, NUMBER_OF_GROUPS, dataset);    
    SVMInterface svm_guy;  
    
    VideoCapture capture(0);
    if(!capture.isOpened()) {
        cout << "No detected camera!" << endl;
        return;
    }
    else {
        cout << "Camera Resolution: " << capture.get(CV_CAP_PROP_FRAME_WIDTH) << "x" << capture.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
        //cout << "Frame Format: " << capture.get(CV_CAP_PROP_FORMAT) << endl;
    }    
    Mat current_frame;
    Mat prev_frame;
    Mat original_frame;
    queue<Mat> frame_queue;
    int queue_num = 1;
	for (unsigned int i = 0; i < GAP_FOR_FRAME_DIFFERENCE; ++i)
	{
		capture >> prev_frame; // ignore first 'GAP_FOR_FRAME_DIFFERENCE' frames.  Read them in and carry on.
		CVT_RE(prev_frame); //RGB?BGR?
		frame_queue.push(prev_frame.clone());
	}
	prev_frame = frame_queue.front();
	frame_queue.pop();
    unsigned int frame_num = GAP_FOR_FRAME_DIFFERENCE - 1;    
    
    queue<Mat> histogram_queue;
    Mat curr_histogram = Mat::zeros(1, NUM_CLUSTERS, CV_32FC1);
    Mat total_histogram = Mat::zeros(1, NUM_CLUSTERS, CV_32FC1);
    int total_N = 0;
    Mat curr_bow = Mat::zeros(1, NUM_CLUSTERS, CV_32FC1);
    svm_model *model = svm_load_model(model_path.c_str());
    svm_node *x = new svm_node[curr_bow.cols+1];  
    double predict_label = 0.0;
    int fps = 30;// capture.get(CV_CAP_PROP_FPS);
    //SurfFeatureDetector *diff_detector = new SurfFeatureDetector(30);
	BRISK *diff_detector = new BRISK(30); 
    string true_act = "Unknown";
    clock_t start_frame, time_frame;
    clock_t start_predict, time_predict;
    start_predict = clock();
    
    while(true) {
        start_frame = clock();
        capture >> current_frame;
        if (current_frame.empty())	
            break;
        original_frame = current_frame;
        CVT_RE(current_frame);
        Mat diff_img(current_frame.rows, current_frame.cols, CV_8U);
        absdiff(current_frame, prev_frame, diff_img);
        vector<KeyPoint> keypoints, diff_keypoints;
        Mat descriptors;
        diff_detector->detect(diff_img, keypoints);
        for(auto keypt = keypoints.begin(); keypt != keypoints.end();) {
            if(!mofreak->sufficientMotion(current_frame, prev_frame, keypt->pt.x, keypt->pt.y, keypt->size))
                keypt=keypoints.erase(keypt);
            else
                ++keypt;
        }        
        mofreak->myFREAKcompute(diff_img, keypoints, descriptors);
		unsigned char *pointer_to_descriptor_row = 0;
		unsigned int keypoint_row = 0;
		for (auto keypt = keypoints.begin(); keypt != keypoints.end(); ++keypt)
		{
			pointer_to_descriptor_row = descriptors.ptr<unsigned char>(keypoint_row);
			int motion = 0;
            //MoFREAKFeature ftr(mofreak->NUMBER_OF_BYTES_FOR_MOTION, mofreak->NUMBER_OF_BYTES_FOR_APPEARANCE);
            //ftr.frame_number = frame_num;
            //ftr.scale = keypt->size;
            //ftr.x = keypt->pt.x;
            //ftr.y = keypt->pt.y;
            Mat feature_vector(1, FEATURE_DIMENSIONALITY, CV_8U);

            for (unsigned i = 0; i < mofreak->NUMBER_OF_BYTES_FOR_APPEARANCE; ++i)
            {
                //ftr.appearance[i] = pointer_to_descriptor_row[i];
                feature_vector.at<unsigned char>(0, i) = pointer_to_descriptor_row[i];
            }

            vector<unsigned int> motion_desc;
            
            mofreak->extractMotionByMotionInterchangePatterns(current_frame, prev_frame, motion_desc, keypt->size, keypt->pt.x, keypt->pt.y);

            for (unsigned i = 0; i < mofreak->NUMBER_OF_BYTES_FOR_MOTION; ++i)
            {
                //ftr.motion[i] = motion_desc[i];
                feature_vector.at<unsigned char>(0, mofreak->NUMBER_OF_BYTES_FOR_APPEARANCE+i) = motion_desc[i];
            }
			keypoint_row++;            
            int best_match = bow_rep.bruteForceMatch(feature_vector);
            curr_histogram.at<float>(0, best_match) += 1;
            total_histogram.at<float>(0, best_match) += 1;
            total_N += 1;
        }
		frame_queue.push(current_frame.clone());
		prev_frame = frame_queue.front();
		frame_queue.pop();
		++frame_num;
        histogram_queue.push(curr_histogram.clone());
        curr_histogram = Mat::zeros(1, NUM_CLUSTERS, CV_32FC1);
        if(histogram_queue.size() > delta_h) {
            Mat temp = histogram_queue.front();
            histogram_queue.pop();
            for(int i=0; i<NUM_CLUSTERS; ++i) {
                total_histogram.at<float>(0, i) -= temp.at<float>(0, i);
                total_N -= (int)temp.at<float>(0, i);
            }
        }
        ++queue_num;
        if(queue_num > delta_f) {
            queue_num = 1;
            for(int i=0; i<NUM_CLUSTERS; ++i)
                curr_bow.at<float>(0, i) = (float)total_histogram.at<float>(0, i) / total_N;                        
            // Compute BOW from MoFREAK and save .bow file
            // clusters.txt is specified in SVM_PATH/clusters.txt by initialization of bow_rep                                 
            for (int col = 0; col < curr_bow.cols; ++col)
            {
                x[col].index = col+1;
                x[col].value = (double)curr_bow.at<float>(0, col);
            }
            x[curr_bow.cols].index = -1;    
            predict_label = svm_predict(model, x);
            time_predict = clock()-start_predict;
            if(keypoints.size() < 20) 
                predict_label = 0; 
            cout << "Frame Number: " << frame_num << " Label: " << labels[predict_label] << "   Time: " << (double)time_predict/CLOCKS_PER_SEC << endl;
            start_predict = clock();
        }
        flip(original_frame, original_frame, 1);
        putText(original_frame, labels[predict_label], cvPoint(10,30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,0), 2);
        imshow("Test Video", original_frame);
        time_frame = clock()-start_frame;
        if(1000/fps-time_frame > 0)
            key = waitKey(1000/fps-time_frame);
        else
            key = waitKey(1);
        if(key==27) break;
	} 
    delete diff_detector;
    delete [] x;
}

void training() {
    string model_path = TRAINING_PATH + "/model.svm";
    string labels_path = TRAINING_PATH + "/labels.txt";
    SVMInterface svm_guy;
    
    vector<string> actions;
    directory_iterator end_iter;
    for (directory_iterator dir_iter(VIDEO_PATH); dir_iter != end_iter; ++dir_iter)
        if(is_directory(dir_iter->status())) 
            actions.push_back(dir_iter->path().filename().generic_string());
    ofstream fout(labels_path);
    fout << actions.size()*100 << " " << actions.size() << " " << "1";
    for(auto it=actions.begin(); it!=actions.end(); ++it)
        fout << endl << *it;
    fout.close();    
    initialize_label();
	NUM_CLUSTERS = actions.size()*100;
	NUM_CLASSES = actions.size();
	NUMBER_OF_GROUPS = 1;
    
    computeMoFREAKFiles();
    cluster();
    computeBOWRepresentation();
    svm_guy.trainModel(SVM_PATH + "/1.train", model_path);
}

void histogram_param(const int delta_f, const int delta_h) {
    const int GAP_FOR_FRAME_DIFFERENCE = 5;
    
    // Initialize file path
    //SVM_PATH = TRAINING_PATH;
    SVM_PATH = "D:/project/action/dataset/KTH/saved/thesis/typical_BRISK30_85/svm/";
    string model_path = SVM_PATH + "/model_18.svm";
    initialize_label();
    string video_filename, video_action, action_video_path, video_file;
    BagOfWordsRepresentation bow_rep(NUM_CLUSTERS, NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES, SVM_PATH, NUMBER_OF_GROUPS, dataset);    
    SVMInterface svm_guy;
    clock_t start, duration=0;
    svm_model *model = svm_load_model(model_path.c_str());
    
    directory_iterator end_iter;
    Mat current_frame;
    Mat prev_frame;
    Mat curr_histogram;
    Mat total_histogram;
    int total_N = 0;
    int total_label=0, correct_label=0;
    int queue_num = 1;
    unsigned int frame_num = GAP_FOR_FRAME_DIFFERENCE - 1;  
    double predict_label = 0.0;
    int action=0, person, video_number;
    int best_match;
    svm_node *x = new svm_node[NUM_CLUSTERS+1];  
    BRISK *diff_detector = new BRISK(30); 
    //cv::SurfFeatureDetector *diff_detector = new cv::SurfFeatureDetector(30);    
    for (directory_iterator dir_iter(VIDEO_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_directory(dir_iter->status()))
		{
			// get folder name.
			video_action = dir_iter->path().filename().generic_string();
			cout << endl << "action: " << video_action << endl;
			mofreak->setCurrentAction(video_action);
			action_video_path = VIDEO_PATH + "/" + video_action;
            total_label=0, correct_label=0;
            duration = 0;
            start = clock();
			for (directory_iterator video_iter(action_video_path); video_iter != end_iter; ++video_iter) {
				if (is_regular_file(video_iter->status())) {
					video_filename = video_iter->path().filename().generic_string();
                    video_file = action_video_path + "/" + video_filename;
                    VideoCapture capture;
                    capture.open(video_file);
                    if (!capture.isOpened())
                        cout << "Could not open file: " << video_filename << endl;
                    
                    queue<Mat> frame_queue;
                    queue_num = 1;
                    for (unsigned int i = 0; i < GAP_FOR_FRAME_DIFFERENCE; ++i)
                    {
                        capture >> prev_frame; // ignore first 'GAP_FOR_FRAME_DIFFERENCE' frames.  Read them in and carry on.
                        CVT_RE(prev_frame);
                        frame_queue.push(prev_frame.clone());
                    }
                    prev_frame = frame_queue.front();
                    frame_queue.pop();
                    frame_num = GAP_FOR_FRAME_DIFFERENCE - 1;    
                    
                    queue<Mat> histogram_queue;
                    curr_histogram = Mat::zeros(1, NUM_CLUSTERS, CV_32FC1);
                    total_histogram = Mat::zeros(1, NUM_CLUSTERS, CV_32FC1);
                    total_N = 0;
                    Mat curr_bow = Mat::zeros(1, NUM_CLUSTERS, CV_32FC1);
                    predict_label = 0.0;
                    mofreak->readMetadata(video_filename, action, video_number, person);
                    while(true) {
                        capture >> current_frame;
                        if (current_frame.empty())
                            break;
                        CVT_RE(current_frame);
                        Mat diff_img(current_frame.rows, current_frame.cols, CV_8U);
                        absdiff(current_frame, prev_frame, diff_img);
                        vector<KeyPoint> keypoints, diff_keypoints;
                        Mat descriptors;
                        diff_detector->detect(diff_img, keypoints);
                        for(auto keypt = keypoints.begin(); keypt != keypoints.end();) {
                            if(!mofreak->sufficientMotion(current_frame, prev_frame, keypt->pt.x, keypt->pt.y, keypt->size))
                                keypt=keypoints.erase(keypt);
                            else
                                ++keypt;
                        }        
                        mofreak->myFREAKcompute(diff_img, keypoints, descriptors);
                        unsigned char *pointer_to_descriptor_row = 0;
                        unsigned int keypoint_row = 0;
                        for (auto keypt = keypoints.begin(); keypt != keypoints.end(); ++keypt)
                        {
                            pointer_to_descriptor_row = descriptors.ptr<unsigned char>(keypoint_row);
                            int motion = 0;
                            MoFREAKFeature ftr(mofreak->NUMBER_OF_BYTES_FOR_MOTION, mofreak->NUMBER_OF_BYTES_FOR_APPEARANCE);
                            ftr.frame_number = frame_num;
                            ftr.scale = keypt->size;
                            ftr.x = keypt->pt.x;
                            ftr.y = keypt->pt.y;
                            Mat feature_vector(1, FEATURE_DIMENSIONALITY, CV_8U);

                            for (unsigned i = 0; i < mofreak->NUMBER_OF_BYTES_FOR_APPEARANCE; ++i)
                            {
                                ftr.appearance[i] = pointer_to_descriptor_row[i];
                                feature_vector.at<unsigned char>(0, i) = pointer_to_descriptor_row[i];
                            }

                            vector<unsigned int> motion_desc;
                            
                            mofreak->extractMotionByMotionInterchangePatterns(current_frame, prev_frame, motion_desc, keypt->size, keypt->pt.x, keypt->pt.y);

                            for (unsigned i = 0; i < mofreak->NUMBER_OF_BYTES_FOR_MOTION; ++i)
                            {
                                ftr.motion[i] = motion_desc[i];
                                feature_vector.at<unsigned char>(0, mofreak->NUMBER_OF_BYTES_FOR_APPEARANCE+i) = motion_desc[i];
                            }

                            ftr.action = action;
                            ftr.video_number = video_number;
                            ftr.person = person;

                            ftr.motion_x = 0;
                            ftr.motion_y = 0;

                            mofreak->features.push_back(ftr);
                            keypoint_row++;
                            
                            best_match = bow_rep.bruteForceMatch(feature_vector);
                            curr_histogram.at<float>(0, best_match) += 1;
                            total_histogram.at<float>(0, best_match) += 1;
                            total_N += 1;
                        }      
                        frame_queue.push(current_frame.clone());
                        prev_frame = frame_queue.front();
                        frame_queue.pop();
                        ++frame_num;
                        histogram_queue.push(curr_histogram.clone());
                        curr_histogram = Mat::zeros(1, NUM_CLUSTERS, CV_32FC1);
                        if(histogram_queue.size() > delta_h) {
                            Mat temp = histogram_queue.front();
                            histogram_queue.pop();
                            for(int i=0; i<NUM_CLUSTERS; ++i) {
                                total_histogram.at<float>(0, i) -= temp.at<float>(0, i);
                                total_N -= (int)temp.at<float>(0, i);
                            }
                        }
                        ++queue_num;
                        if(queue_num > delta_f) {
                            queue_num = 1;
                            for(int i=0; i<NUM_CLUSTERS; ++i)
                                curr_bow.at<float>(0, i) = (float)total_histogram.at<float>(0, i) / total_N;
                            // clusters.txt is specified in SVM_PATH/clusters.txt by initialization of bow_rep                                 
                            for (int col = 0; col < curr_bow.cols; ++col)
                            {
                                x[col].index = col+1;
                                x[col].value = (double)curr_bow.at<float>(0, col);
                            }
                            x[curr_bow.cols].index = -1;    
                            predict_label = svm_predict(model, x);
                            total_label++;
                            if((int)predict_label-1 == action || keypoints.empty()) correct_label++;
                        }
                    }          
                }
            }
            cout << "Total: " << total_label << " Correct: " << correct_label << " Accuracy: " << (double)correct_label/total_label << endl;                    
            cout << "duration: " << (double)(clock()-start)/CLOCKS_PER_SEC << " seconds" << endl;
		}
	}          
    delete [] x;
    delete diff_detector;
}

void convert_one_mofreak(BagOfWordsRepresentation &bow_rep, string input, string output) {
    bool success;
    Mat bow_feature;
    try
    {
        bow_feature = bow_rep.buildHistogram(input, success);
    }
    catch (cv::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        exit(1);
    }
    if (!success)
    {
        std::cout << "Bag-of-words feature construction was unsuccessful.  Investigate." << std::endl;
        exit(1);
    }         
    stringstream ss;
    ss << (0) << " "; // label for svm
    for (int col = 0; col < bow_feature.cols; ++col)
    {
        ss << (int)(col + 1) << ":" << (float)bow_feature.at<float>(0, col) << " ";
    }
    string current_line;
    current_line = ss.str();
    ss.str("");
    ss.clear();
    
    ofstream fout;
    fout.open(output);
    fout << current_line << endl;
    fout.close(); 
}

void convert_bow() {
    BagOfWordsRepresentation bow_rep(NUM_CLUSTERS, NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES, SVM_PATH, NUMBER_OF_GROUPS, dataset);   
	directory_iterator end_iter;
	cout << "Here are the mofreak files: " << MOFREAK_PATH << endl;
	cout << "BOW files will go here: " << BOW_PATH << endl;
	for (directory_iterator dir_iter(MOFREAK_PATH); dir_iter != end_iter; ++dir_iter)
	{
        if (is_directory(dir_iter->status()))
		{
			string video_action = dir_iter->path().filename().generic_string();
			cout << "action: " << video_action << endl;
			string action_video_path = MOFREAK_PATH + "/" + video_action;
			for (directory_iterator video_iter(action_video_path); video_iter != end_iter; ++video_iter)
			{
				if (is_regular_file(video_iter->status()))
				{
					string video_filename = video_iter->path().filename().generic_string();
					if (video_filename.substr(video_filename.length() - 7, 7) == "mofreak")
					{
						string bow_path = BOW_PATH + "/" + video_action + "/" + video_filename + ".bow";
						boost::filesystem::path dir_to_create(BOW_PATH + "/" + video_action + "/");
						boost::system::error_code returned_error;
						boost::filesystem::create_directories(dir_to_create, returned_error);
						if (returned_error)
						{
							std::cout << "Could not make directory " << dir_to_create.string() << std::endl;
							exit(1);
						}	
                        convert_one_mofreak(bow_rep, video_iter->path().generic_string(), bow_path);
					}
				}
			}
		}
	}    
}

vector<string> split_file(const string &s, char delim) {
    vector<std::string> elems;
    stringstream ss(s);
    string item;
    while(getline(ss, item, delim)) 
        elems.push_back(item);
    return elems;
}

void bow_to_svm() {
    BagOfWordsRepresentation bow_rep(NUM_CLUSTERS, NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES, SVM_PATH, NUMBER_OF_GROUPS, dataset);   
    bow_rep.intializeBOWMemory(SVM_PATH);	// ensure that we have the correct number of open files
    cout << "Here are the bow files: " << BOW_PATH << endl;
	cout << "SVM files will go here: " << SVM_PATH << endl;
    directory_iterator end_iter;
    int act = 0;
    ifstream fin;
	for (directory_iterator dir_iter(BOW_PATH); dir_iter != end_iter; ++dir_iter)
	{
        if (is_directory(dir_iter->status()))
		{
			string video_action = dir_iter->path().filename().generic_string();
			cout << "action: " << video_action << endl;
			string action_video_path = BOW_PATH + "/" + video_action;
            ++act;
			for (directory_iterator video_iter(action_video_path); video_iter != end_iter; ++video_iter)
			{
				if (is_regular_file(video_iter->status()))
				{
					string bow_filename = video_iter->path().filename().generic_string();
					if (bow_filename.substr(bow_filename.length() - 3, 3) == "bow")
					{
                        fin.open(BOW_PATH + "/" + video_action + "/" + bow_filename);
                        fin.seekg(0, fin.end);
                        long size = fin.tellg();
                        fin.seekg(2);
                        char* buf = new char[size+2];
                        fin.read(buf+4, size-2);
                        fin.close();
                        buf[0] = (act>=100)? '1': ' ';
                        buf[1] = (act>=10)? '0'+(act/10)%10: ' ';
                        buf[2] = '0'+act%10;
                        buf[3] = ' ';
                        vector<string> filename_parts = split_file(bow_filename, '_');
                        int group = 0;
                        stringstream(filename_parts[2].substr(filename_parts[2].length()- 2,2)) >> group;
                        for(int i=0; i<NUMBER_OF_GROUPS; ++i)
                            if(i == group-1)
                                (*bow_rep.testing_files[i]).write(buf, size+1);
                            else
                                (*bow_rep.training_files[i]).write(buf, size+1);                                
                        delete [] buf;
					}
				}
			}
		}
	}       
    /*
	// for each group, write the training and testing cross-validation files.
	for (int i = 0; i < NUMBER_OF_GROUPS; ++i)
	{
		cout << "number of training features: " << bow_training_crossvalidation_sets[i].size() << endl;
		for (unsigned line = 0; line < bow_training_crossvalidation_sets[i].size(); ++line)
		{
			try
			{
				*training_files[i] << bow_training_crossvalidation_sets[i][line] << endl;
			}
			catch (exception &e)
			{
				cout << "Error: " << e.what() << endl;
				exit(1);
			}
		}
		
		cout << "number of testing features: " << bow_testing_crossvalidation_sets[i].size() << endl;
		for (unsigned line = 0; line < bow_testing_crossvalidation_sets[i].size(); ++line)
		{
			try
			{
				*testing_files[i] << bow_testing_crossvalidation_sets[i][line] << endl;
			}
			catch (exception &e)
			{
				cout << "Error: " << e.what() << endl;
				exit(1);
			}
		}
	}
    */
	cout << "Finished writing to cross-validation files." << endl;

	// close the libsvm training and testing files.
	for (int i = 0; i < NUMBER_OF_GROUPS; ++i)
	{
		bow_rep.training_files[i]->close();
		bow_rep.testing_files[i]->close();

		delete bow_rep.training_files[i];
		delete bow_rep.testing_files[i];
	}

	cout << "Closed all cross-validation files. " << endl;

	bow_rep.training_files.clear();
	bow_rep.testing_files.clear();
    
}

void main(int argc, char *argv[])
{
	setParameters();
	clock_t start, end;
    start = end = clock();
	mofreak = new MoFREAKUtilities(dataset);

	if (state == DETECT_MOFREAK)
	{
		start = clock();
		computeMoFREAKFiles();
		end = clock();
	}

	// This is the most commonly used scenario.
	// Compute MoFREAK descriptors across the dataset,
	// cluster them,
	// compute the bag-of-words representation,
	// and classify.
	else if (state == DETECTION_TO_CLASSIFICATION)
	{
		start = clock();
		computeMoFREAKFiles();
		end = clock();
		cout << "#MoFREAK construction: " << (end - start)/(double)CLOCKS_PER_SEC << " seconds! " << endl << endl;

		start = clock();

		if (dataset == TRECVID)
		{
			pickClusters();
			computeBOWHistograms(false);
			computeSVMResponses();
			detectEvents();
		}
		
		else if (dataset == KTH || dataset == UTI2 || dataset == UCF101 || dataset == WEIZMANN)
		{
			clock_t start1 = clock();
			cluster();
			clock_t end1 = clock();
			cout << "#clustering: " << (end1 - start1)/(double)CLOCKS_PER_SEC << " seconds! " << endl << endl;
			
			start1 = clock();
			computeBOWRepresentation();
			end1 = clock();
			cout << "#compute BOW: " << (end1 - start1)/(double)CLOCKS_PER_SEC << " seconds! " << endl << endl;

			
			start1 = clock();
			double avg_acc = classify();
			end1 = clock();
			cout << "#classify: " << (end1 - start1)/(double)CLOCKS_PER_SEC << " seconds! " << endl << endl;
		}

		else if (dataset == HMDB51)
		{
			clusterHMDB51();
			computeBOWRepresentation();
			classify();
		}

		cout << "deleting mofreak..." << endl;
		delete mofreak;
		cout << "deleted" << endl;
		end = clock();
	}
    else if (state == CLASSIFICATION) {
        clock_t start1 = clock();
        cluster();
        clock_t end1 = clock();
        cout << "#clustering: " << (end1 - start1)/(double)CLOCKS_PER_SEC << " seconds! " << endl << endl;
        
        start1 = clock();
        computeBOWRepresentation();
        end1 = clock();
        cout << "#compute BOW: " << (end1 - start1)/(double)CLOCKS_PER_SEC << " seconds! " << endl << endl;

        
        start1 = clock();
        double avg_acc = classify();
        end1 = clock();
        cout << "#classify: " << (end1 - start1)/(double)CLOCKS_PER_SEC << " seconds! " << endl << endl;    
    }
    else if (state == RECOGNITION)
    {
		start = clock();
        if(argc != 2) {
            cout << "Usage: " << argv[0] << " your_video_file" << endl;
            return;
        }
        recognition(argv[1]);
		end = clock();
    }
    else if (state == RECOGNITION_ONLINE)
    {
        if(argc < 4 || argc > 6) {
            cout << "Usage: " << argv[0] << " your_video_file delta_h delta_f [draw_histogram=1] [play_video=1]" << endl;
            return;
        }
        if(argc > 4 && atoi(argv[4])==0) draw_histogram = false;
        if(argc > 5 && atoi(argv[5])==0) play_video = false;
		start = clock();
        recognition_online(argv[1], atoi(argv[2]), atoi(argv[3]));
		end = clock();
    }
    else if (state == VIDEO_ONLINE)
    {
		start = clock();
        video_online();
		end = clock();
    }
    else if (state == TRAINING)
    {
        start = clock();
        training();
        end = clock();
    }
    else if (state == HISTOGRAM_PARAM)
    {
		start = clock();
        if(argc != 3) {
            cout << "Usage: " << argv[0] << " delta_h delta_f " << endl;
            return;
        }
        histogram_param(atoi(argv[1]), atoi(argv[2]));
		end = clock();
    }
    else if (state == CONVERT_BOW)
    {
        start = clock();
        convert_bow();
        end = clock();
    }
    else if (state == BOW_TO_SVM)
    {
        start = clock();
        bow_to_svm();
        end = clock();        
    }
	// TRECVID cases
	else if (state == PICK_CLUSTERS)
	{
		start = clock();
		pickClusters();
		end = clock();
	}
	else if (state == COMPUTE_BOW_HISTOGRAMS)
	{
		start = clock();
		const bool POSITIVE_EXAMPLES = false;
		computeBOWHistograms(POSITIVE_EXAMPLES);
		end = clock();
	}
	else if (state == DETECT)
	{
		start = clock();
		detectEvents();
		end = clock();
	}

	else if (state == TRAIN)
	{
		start = clock();
		trainTRECVID();
		end = clock();
	}

	else if (state == GET_SVM_RESPONSES)
	{
		start = clock();
		computeSVMResponses();
		end = clock();
	}

	cout << "Took this long: " << (end - start)/(double)CLOCKS_PER_SEC << " seconds! " << endl;
	cout << "All done.  Press any key to continue..." << endl;
	cout << "Dumping memory leak info" << endl;
	_CrtDumpMemoryLeaks();
}