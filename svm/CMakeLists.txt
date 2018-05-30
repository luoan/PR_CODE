cmake_minimum_required( VERSION 2.8)

project( svm)

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
add_executable( svm main.cpp)


#add_library( tostring tostring.cpp)
#add_library( read_tag read_tag.cpp) # wrong since read_tag need tostring
#target_link_libraries(knn   tostring read_tag  ${OpenCV_LIBS})


add_library( luoan_lib tostring.cpp read_data.cpp)

target_link_libraries( svm luoan_lib  ${OpenCV_LIBS})

