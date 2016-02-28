MoFREAK_Hardware
================

For ICASSP version, please refer to ICASSP branch.

The system is originally designed for hardware implementation of action recognition. The algorithm is originated from Chris Whiten (https://github.com/ChrisWhiten/MoFREAK), and is improved by Chung-Yu Chi for the DSPIC Research Lab at National Taiwan University GIEE. Work was tested for KTH and Weizmann dataset to prove the robustness.

Questions can be forwarded to Chung-Yu Chi at craig@video.ee.ntu.edu.tw


Dependencies:
------------------
- Boost 1.50.0 (http://www.boost.org/)
- OpenCV 2.4.6 (http://opencv.org)
- Built with Visual Studio 2010

Usage:
-----------------
First of all, the file path of dataset should be correctly created. An example for KTH dataset is like:

```
KTH ------ original ------ boxing
     |               |---- handclapping
     |               |---- handwaving
     |               |---- jogging
     |               |---- running
     |               |---- walking
     |---- mofreak  ------ (empty)
     |---- svm      ------ (empty)
```

Then the file pathes in setParameters() should be modified to match above directories.

Second, the file pathes in the constructor of MoFREAKUtilities (at the top of MoFREAKUtilities.cpp) should be modified to the absolute location of the project. There are total 4 files to be read in initially.


To run the system, we can modify the global parameters to achieve different function. The constructs for performing action recognition already exist for some datasets.  Within main.cpp, there is a setParameters() function that outlines the file structure required for each dataset.  Dataset can be selected at the top of main.cpp with the "dataset" variable, selected from the "datasets" enum.

To exclusively compute the MoFREAK features across a dataset, set the "state" variable at the top of main.cpp to "DETECT_MOFREAK".  This will process each video file, creating a .mofreak file containing its descriptors.

To compute MoFREAK features across the dataset and perform the entire recognition pipeline, set the "state" variable to "DETECTION_TO_CLASSIFICATION".  This will compute the MoFREAK files, cluster the features and compute a bag-of-words representation.  Finally, it will take the bag-of-words representations and classify them with an SVM.

The run time of each stage is defaultly printed on standard output.


Feature Format: (From Chris Whiten)
----------------
Within a .mofreak file, each row consists of a single feature.  That feature is organized as follows:
- [x location] [y location] [frame number] [scale] [throw-away] [throw-away] [8 bytes of appearance data] [8 bytes of motion data]

The x and y location, as well as the scale, are floating point numbers.  The frame number is an integer.  The final 16 bytes of descriptor data are unsigned integers (1 byte per integer).  The throw-away values are floating point values that should always be 0.  They are simply artifacts from previous iterations of the descriptor, and should be ignored.
