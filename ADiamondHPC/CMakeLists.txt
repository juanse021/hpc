cmake_minimum_required(VERSION 2.8)
project( Adiamond )
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=sm_52)
cuda_add_executable( Adiamond Adiamond.cu )
target_link_libraries( Adiamond ${OpenCV_LIBS} )