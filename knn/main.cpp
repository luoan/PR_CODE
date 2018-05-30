#include "read_data.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include <stdio.h>

int main()
{
  cv::Mat train_data, train_labels;
  cv::Mat test_data, test_labels;

  read_train(train_data, train_labels);
  read_test(test_data, test_labels);
  
  //int k = 3;
  cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(train_data, cv::ml::ROW_SAMPLE, train_labels);

  cv::Ptr<cv::ml::KNearest> model = cv::ml::KNearest::create();  
  //model->setDefaultK(k);
  model->setIsClassifier(true);
  model->train(tData);

  std::cout << "k" << "\t" << "pos_accuracy\t" <<  "neg_accuracy" << std::endl;
  for (int i = 1; i != 20; ++i) {
    
    std::cout << i << "\t";
    cv::Mat result;// = cv::Mat::zeros(6875, 1, CV_32FC1);
    double pos_count = 0;
    double neg_count = 0;
    
    model->findNearest(test_data, i, result);
    //std::cout << result.elemSize1() << std::endl;
    
    for (int j = 0; j != 2043; ++j) {
      if (std::abs(result.at<float>(j, 0) - test_labels.at<int>(j)) <= FLT_EPSILON)
        pos_count++;
    }
    printf("%f\t",  pos_count / 2043);

    for (int j = 2043; j != 6875; ++j) {
      if (std::abs(result.at<float>(j, 0) - test_labels.at<int>(j)) <= FLT_EPSILON)
        neg_count++;
    }
    
    printf("%f\n",  neg_count / 4832);
    
  }

  return 0;
  /*
  double grades = 0;
  for (int i = 0; i != 2043; ++i) {
    cv::Mat result;
    double f = model->findNearest(test_data.row(1), 1, result);
    f = std::abs(f - test_labels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
    grades += f;
  }
  printf("%f\n", grades/2043);
  */
  
  /*
  // for test pos
  double pos_right = 0;
  for (int i = 0; i != 2043; ++i) {
    //cv::Mat sample = test_data.row(i);
    // use predict accuracy i<2043 model-predict 
    // k=1 0.209496
    // k=2 0.314733
    // k=3 0.210475
    // k=4 0.273128
    // k=5 0.200196
    double r = model->predict(test_data.row(i));
    std::cout << r << " ";
    r = std::abs(r - test_labels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
    pos_right += r;
  }
  printf("%f\n", pos_right / 2043);
  

  
  // for test neg
  double neg_right = 0;
  for (int i =2043; i != 6875; ++i) {
    //k=1 0.968543
    //k=2 0.940604
    //k=3 0.968129
    //k=4 0.952401
    //k=5 0.969371
    double r = model->predict(test_data.row(i));
    std::cout << r << " ";
    r = std::abs(r - test_labels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
    neg_right += r;
  }
  printf("%f\n", neg_right / 4832);
  */

}
/*
int main()
{
  std::ofstream outfile;
  std::string true_pos = "../train/pos/";
  outfile.open("../test_out.txt");

  cv::Mat data, labels;
  for (int i = 0; i != 3605; ++i) {
    
    cv::Mat tmp;
    
    std::string true_pos_count = true_pos + int2string(i)+ ".txt";
    //outfile << true_pos_count << "\n";

  }
  return 0;
}
*/
/*
int main()
{
  std::fstream infile;
  infile.open("./train/pos/0.txt");
  
  cv::Mat data = cv::Mat::zeros(2330, 1,CV_32FC1);
  
  for (int i = 0; i != 2330; ++i) {
    infile >> data.at<float>(i, 0);
  }
  
  
  return 0;
}*/
