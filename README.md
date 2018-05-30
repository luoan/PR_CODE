# PR_CODE
将这三个文件夹放在PR_dataset目录内，与test, train在同一目录下

CentOS 7.5  cmake  2.8  make   3.82

代码是C++语言编写的
使用cmake编译OpenCV程序
比如要测试SVM分类器
cd svm  

mkdir build

cd build

cmake ..

make

./svm

修改参数，直接修改相应源代码即可，修改完成在build目录下重复
cmake ..

make

./svm
