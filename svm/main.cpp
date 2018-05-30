#include "read_data.h"

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <algorithm>
#include <stdio.h>

int main()
{
  cv::Mat train_data, train_labels;
  cv::Mat test_data, test_labels;

  read_train(train_data, train_labels);
  read_test(test_data, test_labels);
  
  
  cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::create();
  model->setType(cv::ml::SVM::C_SVC);
  model->setC(0.008);
  //model->setClassWeights();
  std::cout << "setc(0.008)" << "\n";
  model->setKernel(cv::ml::SVM::LINEAR);//(cv::ml::SVM::LINEAR);
  model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, (int)1e7, 1e-6));  
  //model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));  
  //model->train(tData);
  model->train(train_data, cv::ml::ROW_SAMPLE, train_labels);
  double r;
  
  double train_pos_count=0, train_neg_count=0;
  for (int i = 0; i != 3605; ++i) {
    r = model->predict(train_data.row(i));   
    r = std::abs(r - train_labels.at<int>(i)) < FLT_EPSILON ? 1.f : 0.f;
    train_pos_count += r;
  }
  printf("accuracy for train data pos: %f\n", train_pos_count / 3605);
  for (int i = 3605; i != 13660; ++i) {
    r = model->predict(train_data.row(i));
    r = std::abs(r - train_labels.at<int>(i)) < FLT_EPSILON ? 1.f : 0.f;
    train_neg_count += r;
  }
  printf("accuracy for train data neg: %f\n\n", train_neg_count / 10055);
  // setc=0.001   pos=0.896255 neg=0.972253
  // default settermcriteria setc=0.005 pos=0.626075 neg=0.830035
  // setc=0.003   pos=0.897642 neg=0.973347
  // setc=0.005   pos=0.899861 neg=0.973247
  // setc=0.008   pos=0.900693 neg=0.972551
  // setc=0.01    pos=0.896533 neg=0.971556
  // setc=0.05    pos=0.712899 neg=0.946196
  // setc=0.1     pos=0.705964 neg=0.936350
  //
  double pos_count = 0;
  double neg_count = 0;
  
  for (int i = 0; i != 2043; ++i) {
    r = model->predict(test_data.row(i));
    r = std::abs(r - test_labels.at<int>(i)) < FLT_EPSILON ? 1.f : 0.f;
    pos_count += r;
  }
  printf("accuracy for pos: %f\n", pos_count / 2043);

  for (int i = 2043; i != 6875; ++i) {
    r = model->predict(test_data.row(i));
    r = std::abs(r - test_labels.at<int>(i)) < FLT_EPSILON ? 1.f : 0.f;
    neg_count += r;
  }
  printf("accuracy for neg: %f\n", neg_count / 4832);
  // linear
  // for test
  // setc=0.001 pos=0.710720  neg=0.926738
  // setc=0.003 pos=0.707783  neg=0.923013
  // setc=0.005 pos=0.708272  neg=0.921565
  // setc=0.008 pos=0.718062  neg=0.922185
  // setc=0.01  pos=0.716593  neg=0.922392
  // setc=0.05  pos=0.573666  neg=0.922599
  // setc=0.1   pos=0.563387  neg=0.911838
  // default settermcriteria run much fast but result is terrible pos=0.550171 neg=0.820571
  //
  // error data!!!!!!!!!!!!
  // setc 0.000001 pos 0.800294 neg 0.722682
  // setc 0.000005 pos 0.845815 neg 0.705298
  // setc 0.00001 pos 0.859520 neg 0.698469
  // setc 0.00005 pos 0.882526 neg 0.693916
  // setc 0.0001 pos 0.887901 neg 0.694536
  // setc 0.0005 pos 0.896721 neg 0.692674
  // setc 0.0008 pos 0.898678 neg 0.691639
  // setc 0.008  pos 0.901126  neg 0.689570
  // setc 0.005  pos 0.902594  neg 0.690397
  // setc 0.003 pos 0.900638   neg 0.691846
  // setc 0.001 pos 0.899168  neg 0.691432
  // setc 0.01 pos 0.898678  neg 0.689983
  //setc 0.05  pos 0.727362  neg 0.739445
  //setc 0.1   pos 0.684288  neg 0.719992
  //setc 0.5   pos 0.663730  neg 0.695571
  //
  //
  //nu_svc pos 0.25 neg 0.32
  //RBF 0 1
  //poly error
  //sigmod error
  return 0;
}
