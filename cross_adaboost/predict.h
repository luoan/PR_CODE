#ifndef _predict_h
#define _predict_h


#include <opencv2/opencv.hpp>

void verify( const cv::Mat& train_data, const cv::Mat& train_labels,
  			 const cv::Mat& verify_data, const cv::Mat& verify_labels);

void predict( const cv::Mat& train_data, const cv::Mat& train_labels,
			  const cv::Mat& test_data,  const cv::Mat& test_labels);


#endif
