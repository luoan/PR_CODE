#include "predict.h"

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>

void verify( const cv::Mat& train_data, const cv::Mat& train_labels,
  			 const cv::Mat& verify_data, const cv::Mat& verify_labels)
{
  cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(train_data, cv::ml::ROW_SAMPLE, train_labels);
  std::vector<double> priors(2);
  priors[0] = 1;
  priors[1] = 26; 
  
  //std::cout << verify_data.rows << std::endl;
  cv::Ptr<cv::ml::Boost> model = cv::ml::Boost::create();
  model->setBoostType(cv::ml::Boost::GENTLE);
  model->setWeakCount(100);
  model->setWeightTrimRate(0.95);
  model->setMaxDepth(2);
  model->setUseSurrogates(false);
  model->setPriors(cv::Mat(priors));
  //std::cout << train_labels_part1.reshape(0, 1) << std::endl;
  model->train(tData);
  
  double train_pos_count = 0, train_neg_count = 0;
  double train_pos_verify = 0;
  double train_neg_verify = 0;

  double r;
  //std::cout << train_data.rows << std::endl;
  for (int i = 0; i != train_data.rows; ++i) {
    r = model->predict(train_data.row(i));
    r = std::abs(r - train_labels.at<int>(i)) < FLT_EPSILON ? 1.f : 0.f;
    if (i < 3000)    
      train_pos_count += r;
    else 
      train_neg_count += r;
  }
  printf("accuracy for train data pos: %f\n", train_pos_count / 3000);
  printf("accuracy for train data neg: %f\n", train_neg_count / (train_data.rows-3000));

 
  for (int i = 0; i != verify_data.rows; ++i) {
    r = model->predict(verify_data.row(i));
    r = std::abs(r - verify_labels.at<int>(i)) < FLT_EPSILON ? 1.f : 0.f;
    if (i < 605)
      train_pos_verify += r;
    else 
      train_neg_verify += r;
  }
  printf("accuracy for train data pos verify: %f\n",   train_pos_verify / 605);
  printf("accuracy for train data neg verify: %f\n\n", train_neg_verify / (verify_data.rows-605));
}


void predict( const cv::Mat& train_data, const cv::Mat& train_labels,
			  const cv::Mat& test_data,  const cv::Mat& test_labels)
{
  cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(train_data, cv::ml::ROW_SAMPLE, train_labels);
  std::vector<double> priors(2);
  priors[0] = 1;
  priors[1] = 26; 
  
  //std::cout << verify_data.rows << std::endl;
  cv::Ptr<cv::ml::Boost> model = cv::ml::Boost::create();
  model->setBoostType(cv::ml::Boost::GENTLE);
  model->setWeakCount(100);
  model->setWeightTrimRate(0.95);
  model->setMaxDepth(3);
  model->setUseSurrogates(false);
  model->setPriors(cv::Mat(priors));
  //std::cout << train_labels_part1.reshape(0, 1) << std::endl;
  model->train(tData);
  
  double test_pos_count = 0;
  double test_neg_count = 0;

  double r;
  //std::cout << train_data.rows << std::endl;
  for (int i = 0; i != test_data.rows; ++i) {
    r = model->predict(test_data.row(i));
    r = std::abs(r - test_labels.at<int>(i)) < FLT_EPSILON ? 1.f : 0.f;
    if (i < 2043)    
      test_pos_count += r;
    else 
      test_neg_count += r;
  }
  printf("accuracy for test data pos: %f\n",   test_pos_count / 2043);
  printf("accuracy for test data neg: %f\n\n", test_neg_count / (test_data.rows-2043));
}

