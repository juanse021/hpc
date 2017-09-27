#include <cuda.h>
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <time.h>
#include <fstream>

__global__ void grayImage(unsigned char *image_begin, int width, int height, unsigned char *image_end) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < height) && (col < width)) {
        image_end[row*width+col] = image_end[(row*width+col) * 3] * 0.3 + \
        image_end[(row*width+col) * 3] * 0.59 + \
        image_end[(row*width+col) * 3] * 0.11;
    }
}

int main(int argc, char **argv) {
    char *imageName = argv[1];
    Mat image = imread(imageName, 1);
    int width = image.size().width();
    int height = image.size().height();

    if (!image.data) {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cudaError_t error = cudaSuccess;
    int size = width * height * 3 * sizeof(unsigned char*);
    unsigned char *h_imageA, *h_imageB, *d_imageA, *d_imageB;

    // Separar memoria de imagen color en host
    h_imageA = (unsigned char*)malloc(size));
    error = cudaMalloc((void**)&d_imageA, sizeof(unsigned char*) * width * height);
    if (error != cudaSuccess) {
        printf("Error.... d_imageA \n");
        return -1;
    }
    // Separar memoria de imagen gris en host
    h_imageB = (unsigned char*)malloc(size));
    error = cudaMalloc((void**)&d_imageB, sizeof(unsigned char*) * width * height);
    if (error != cudaSuccess) {
        printf("Error.... d_imageB \n");
        return -1;
    }

    h_imageA = image.data;

    error = cudaMemcpy(d_imageA, h_imageA, size,cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error... h_imageA a d_imageA \n");
        return -1;
    }

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(ceil(width/float(32)), ceil(height/float(32)), 1);
    grayImage<<<dimGrid, dimBlock>>>(d_imageA, width, height, d_imageB);
    cudaDeviceSynchronize();

    error = cudaMemcpy(h_imageB, d_imageB, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error... d_imageB a h_imageB \n");
        return -1;
    }

    Mat imageGray;
    imageGray.create(width, heigth, CV_8UC1);
    imageGray.data = d_imageB;



    return 0;
}
