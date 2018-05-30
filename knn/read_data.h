#ifndef _read_data_h
#define _read_data_h


#include "tostring.h"

#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

void read_train(cv::Mat& train_data, cv::Mat& train_labels );
void read_test (cv::Mat& test_data,  cv::Mat& test_labels );

#endif
