将这三个文件夹放在PR_dataset目录内，与test, train在同一目录下

我的环境
CentOS 7.5
cmake 2.8.12
make 3.82
g++ 4.8.5
opencv-3.4.1

代码是C++语言编写的 使用cmake编译OpenCV程序 比如要测试SVM分类器
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

如果使用g++命令编译程序
read_data.cpp需要修改数据路径，或者在当前目录下mkdir一个新目录，cd进去，将a.out拷贝进去执行
g++ main.cpp read_data.h read_data.cpp tostring.h tostring.cpp `pkg-config --libs --cflags opencv`
./a.out
