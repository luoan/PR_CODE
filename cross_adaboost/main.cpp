#include "read_data.h"
#include "predict.h"

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <algorithm>
#include <vector>

#include <stdio.h>

int main()
{
  cv::Mat train_data, train_labels;
  cv::Mat test_data, test_labels;

  read_train(train_data, train_labels);
  read_test(test_data, test_labels);
  
  //0-3000   3605-111980
  cv::Mat train_data_part1 = train_data.rowRange( 0, 3000);
  train_data_part1.push_back(train_data.rowRange( 3605, 11980));
  
  cv::Mat train_labels_part1 = train_labels.rowRange( 0, 3000);
  train_labels_part1.push_back(train_labels.rowRange( 3605, 11980));
  
  cv::Mat train_data_verify1 = train_data.rowRange(3000, 3605);
  train_data_verify1.push_back(train_data.rowRange(11980, 13660));

  cv::Mat train_labels_verify1 = train_labels.rowRange(3000, 3605);
  train_labels_verify1.push_back(train_labels.rowRange(11980, 13660));


  //0-2400+3005-3605    3605-10305 11985-13660
  cv::Mat train_data_part2 =  train_data.rowRange( 0, 2400);
  train_data_part2.push_back( train_data.rowRange( 3005, 3605));
  train_data_part2.push_back( train_data.rowRange( 3605, 10305));
  train_data_part2.push_back( train_data.rowRange( 11985, 13660));
  
  cv::Mat train_labels_part2 =  train_labels.rowRange( 0, 2400);
  train_labels_part2.push_back( train_labels.rowRange( 3005, 3605));
  train_labels_part2.push_back( train_labels.rowRange( 3605, 10305));
  train_labels_part2.push_back( train_labels.rowRange( 11985, 13660));
  
  cv::Mat train_data_verify2 = train_data.rowRange(2400, 3605);
  train_data_verify2.push_back(train_data.rowRange(10305, 11985));

  cv::Mat train_labels_verify2 = train_labels.rowRange(2400, 3605);
  train_labels_verify2.push_back(train_labels.rowRange(10305, 11985));
  //cv::Mat train_data_verify2 = train_data.rowRange(2400, 3005);
  //cv::Mat train_labels_verify2 = train_labels.rowRange(10305, 11985);


  //0-1800 2400-3605     3605-8630 10305-13660
  cv::Mat train_data_part3 =  train_data.rowRange( 0, 1800);
  train_data_part3.push_back( train_data.rowRange( 2405, 3605));
  train_data_part3.push_back( train_data.rowRange( 3605, 8630));
  train_data_part3.push_back( train_data.rowRange( 10305,13660));
  
  cv::Mat train_labels_part3 =  train_labels.rowRange( 0, 1800);
  train_labels_part3.push_back( train_labels.rowRange( 2405, 3605));
  train_labels_part3.push_back( train_labels.rowRange( 3605, 8630));
  train_labels_part3.push_back( train_labels.rowRange( 10305,13660));
  
  cv::Mat train_data_verify3 = train_data.rowRange(1800, 2405);
  train_data_verify3.push_back(train_data.rowRange(8630, 10305));

  cv::Mat train_labels_verify3 = train_labels.rowRange(1800, 2405);
  train_labels_verify3.push_back(train_labels.rowRange(8630, 10305));
  //cv::Mat train_data_verify3 = train_data.rowRange( 1800, 2405);
  //cv::Mat train_labels_verify3 = train_labels.rowRange(8630, 10305);


  //0-1200 1800-3000     3605-6956 8630-13660
  cv::Mat train_data_part4 =  train_data.rowRange( 0, 1200);
  train_data_part4.push_back( train_data.rowRange( 1805, 3605));
  train_data_part4.push_back( train_data.rowRange( 3605, 6956));
  train_data_part4.push_back( train_data.rowRange( 8630, 13660));
  
  cv::Mat train_labels_part4 =  train_labels.rowRange( 0, 1200);
  train_labels_part4.push_back( train_labels.rowRange( 1805, 3605));
  train_labels_part4.push_back( train_labels.rowRange( 3605, 6956));
  train_labels_part4.push_back( train_labels.rowRange( 8630, 13660));

  cv::Mat train_data_verify4 = train_data.rowRange(1200, 1805);
  train_data_verify4.push_back(train_data.rowRange(6956, 8630));

  cv::Mat train_labels_verify4 = train_labels.rowRange(1200, 1805);
  train_labels_verify4.push_back(train_labels.rowRange(6956, 8630));
  //cv::Mat train_data_verify4 = train_data.rowRange(1200, 1805);
  //cv::Mat train_labels_verify4 = train_labels.rowRange(6956, 8630);


  //0-600 1200-3605      3605-5280 6956-13660
  cv::Mat train_data_part5 =  train_data.rowRange( 0, 600);
  train_data_part5.push_back( train_data.rowRange( 1205, 3605));
  train_data_part5.push_back( train_data.rowRange( 3605, 5280));
  train_data_part5.push_back( train_data.rowRange( 6956, 13660));
  
  cv::Mat train_labels_part5 =  train_labels.rowRange( 0, 600);
  train_labels_part5.push_back( train_labels.rowRange( 1205, 3605));
  train_labels_part5.push_back( train_labels.rowRange( 3605, 5280));
  train_labels_part5.push_back( train_labels.rowRange( 6956, 13660));
  
  cv::Mat train_data_verify5 = train_data.rowRange(600, 1205);
  train_data_verify5.push_back(train_data.rowRange(5280, 6956));

  cv::Mat train_labels_verify5 = train_labels.rowRange(600, 1205);
  train_labels_verify5.push_back(train_labels.rowRange(5280, 6956));
  //cv::Mat train_data_verify5 = train_data.rowRange(600, 1205);
  //cv::Mat train_labels_verify5 = train_labels.rowRange(5280, 6956);
  

//600-3605   5280-13660
  cv::Mat train_data_part6 = train_data.rowRange( 605, 3605);
  train_data_part6.push_back(train_data.rowRange( 5280, 13660));
  
  cv::Mat train_labels_part6 = train_labels.rowRange( 605, 3605);
  train_labels_part6.push_back(train_labels.rowRange( 5280, 13660));
  
  cv::Mat train_data_verify6 = train_data.rowRange(0, 605);
  train_data_verify6.push_back(train_data.rowRange(3605, 5280));

  cv::Mat train_labels_verify6 = train_labels.rowRange(0, 605);
  train_labels_verify6.push_back(train_labels.rowRange(3605, 5280));
  //cv::Mat train_data_verify6 = train_data.rowRange(0, 605);
  //cv::Mat train_labels_verify6 = train_labels.rowRange(3605, 5280);

  /*
  std::cout << "part1" << std::endl;
  verify(train_data_part1, train_labels_part1, train_data_verify1, train_labels_verify1);

  std::cout << "part2" << std::endl;
  verify(train_data_part2, train_labels_part2, train_data_verify2, train_labels_verify2);
  
  std::cout << "part3" << std::endl;
  verify(train_data_part3, train_labels_part3, train_data_verify3, train_labels_verify3);

  std::cout << "part4" << std::endl;
  verify(train_data_part4, train_labels_part4, train_data_verify4, train_labels_verify4);

  std::cout << "part5" << std::endl;
  verify(train_data_part5, train_labels_part5, train_data_verify5, train_labels_verify5);

  std::cout << "part6" << std::endl;
  verify(train_data_part6, train_labels_part6, train_data_verify6, train_labels_verify6);
  */

  std::cout << "test" << std::endl;
  predict(train_data_part2, train_labels_part2, test_data, test_labels);
}
