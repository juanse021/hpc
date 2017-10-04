#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <malloc.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void grayImageDevice(unsigned char *image_begin, int width, int height, unsigned char *image_end) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < height) && (col < width)) {
        image_end[row*width+col] = image_begin[(row*width+col) * 3 + 2] * 0.3 + image_begin[(row*width+col) * 3 + 1] * 0.59 + image_begin[(row*width+col) * 3] * 0.11;
    }
}

int main(int argc, char **argv) {
    char *imageName = argv[1];
    Mat image = imread(imageName, 1);

    if (!image.data) {
        printf("Could not open or find the image \n");
        return -1;
    }

    Size s = image.size();
    int width = s.width;
    int height = s.height;

    cudaError_t error = cudaSuccess;
    int size = width * height * sizeof(unsigned char) * image.channels();
    int sizeGray = width * height * sizeof(unsigned char);


    unsigned char *h_imageA, *h_imageB, *d_imageA, *d_imageB;

    // h_imageA = (unsigned char*)malloc(size);
    error = cudaMalloc((void**)&d_imageA, size);
    if (error != cudaSuccess) {
        printf("Error.... d_imageA \n");
        return -1;
    }

    h_imageA = image.data;

    error = cudaMemcpy(d_imageA, h_imageA, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error... h_imageA a d_imageA \n");
        return -1;
    }

    // Separar memoria de imagen gris en host
    h_imageB = (unsigned char*)malloc(sizeGray);
    error = cudaMalloc((void**)&d_imageB, sizeGray);
    if (error != cudaSuccess) {
        printf("Error.... d_imageB \n");
        return -1;
    }

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(ceil(height/32.0), ceil(width/32.0), 1);

    grayImageDevice<<<dimGrid, dimBlock>>>(d_imageA, width, height, d_imageB);
    cudaDeviceSynchronize();

    error = cudaMemcpy(h_imageB, d_imageB, sizeGray, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error... d_imageB a h_imageB \n");
        return -1;
    }

    //grayImageHost(h_imageA, width, height, h_imageB);

    Mat imageGray;
    imageGray.create(height, width, CV_8UC1);
    imageGray.data = h_imageB;

    imwrite("ferrari_gray.jpg", imageGray);


    free(h_imageB);
    cudaFree(d_imageA); cudaFree(d_imageB);

    return 0;
}
