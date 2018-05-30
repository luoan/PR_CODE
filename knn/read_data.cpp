#include "read_data.h"
#include "tostring.h"

#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

void read_train(cv::Mat& train_data, cv::Mat& train_labels)
{
  std::fstream infile;
  const std::string pos_position = "../../train/pos/";
  const std::string neg_position = "../../train/neg/";

  cv::Mat tmp = cv::Mat::zeros(1, 2330, CV_32FC1);

  for (int i = 0; i != 3605; ++i) {
    std::string position = pos_position + int2string(i) + ".txt";
    
    infile.open(position.c_str());
    for (int j = 0; j != 2330; ++j) {
      infile >> tmp.at<float>(0, j);  
    }
    infile.close();

    train_data.push_back(tmp);
    train_labels.push_back(0);
  }
  
  for (int i = 0; i != 10055; ++i) {
    const std::string position = neg_position + int2string(i) + ".txt";

    infile.open(position.c_str());
    for (int j = 0; j != 2330; ++j) {
      infile >> tmp.at<float>(0, j);
    }
    infile.close();

    train_data.push_back(tmp);
    train_labels.push_back(1);
  }
}

void read_test(cv::Mat& test_data, cv::Mat& test_labels)
{
  std::fstream infile;
  const std::string pos_position = "../../test/pos/";
  const std::string neg_position = "../../test/neg/";

  cv::Mat tmp = cv::Mat::zeros(1, 2330, CV_32FC1);

  for (int i = 0; i != 2043; ++i) {
    //cv::Mat tmp = cv::Mat::zeros(1, 2330, CV_32FC1);
    std::string position = pos_position + int2string(i) + ".txt";
    
    infile.open(position.c_str());
    for (int j = 0; j != 2330; ++j) {
      infile >> tmp.at<float>(0, j);  
    }
    infile.close();

    test_data.push_back(tmp);
    test_labels.push_back(0);
  }
  
  for (int i = 0; i != 4832; ++i) {
    const std::string position = neg_position + int2string(i) + ".txt";

    infile.open(position.c_str());
    for (int j = 0; j != 2330; ++j) {
      infile >> tmp.at<float>(0, j);
    }
    infile.close();

    test_data.push_back(tmp);
    test_labels.push_back(1);
  }
}

