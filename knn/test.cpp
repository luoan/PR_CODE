#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>

int main()
{
  std::fstream infile;
  infile.open("../train/pos/0.txt");

  cv::Mat mat;

  cv::Mat data = cv::Mat::zeros(1, 2330, CV_32FC1);

  for (int i = 0; i != 2330; ++i) {
    infile >> data.at<float>(0 , i);
  }
  
  mat.push_back(data);
  std::cout << mat.size() << std::endl;
  infile.close();
  infile.open("../train/pos/1.txt");

  for (int i =0; i != 2330; ++i) {
    infile >> data.at<float>(0, i);
  }
  mat.push_back(data);
  //std::cout << mat.size() << std::endl;
  std::cout << mat << std::endl;
 
  return 0;
}
