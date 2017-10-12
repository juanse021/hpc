#include <cuda.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <malloc.h>
#include <opencv2/opencv.hpp>

typedef unsigned char uchar;

using namespace cv;

__global__ void grayImageDevice(const uchar *imgInput, const int width, const int height, uchar *imgOutput) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        imgOutput[row*width+col] = imgInput[(row*width+col) * 3 + 2] * 0.3 + \
                                   imgInput[(row*width+col) * 3 + 1] * 0.59 + \
                                   imgInput[(row*width+col) * 3] * 0.11;
    }
}

__device__ double clamp(double value) {
    if (value > 255)
        value = 255;
    else if (value < 0)
        value = 0;

    return value;
}


__global__ void sobelFilter(const uchar *imgInput, const int width, const int height, uchar *imgOutput) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int sobel_x = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int sobel_y = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    double magnitude_x = 0;
    double magnitude_y = 0;

    const int maskWidth = 3;

    if (col <= 0 || col > width || row <= 0 || row > height)
        return;

    for (int i = 0; i < maskWidth; i++) {
        for (int j = 0; j < maskWidth; j++) {
            int focus_x = i + col;
            int focus_y = j + row;
            magnitude_x += imgOutput[focus_x + focus_y * width] * sobel_x[i * maskWidth + j];
            magnitude_y += imgOutput[focus_x + focus_y * width] * sobel_y[i * maskWidth + j];
        }
    }

    double magnitude = sqrt(magnitude_x * magnitude_x + magnitude_y * magnitude_y);
    imgOutput[row + col * width] = clamp(magnitude);

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
    int size = width * height * sizeof(uchar) * image.channels();
    int sizeGray = width * height * sizeof(uchar);

    uchar *h_imageA, *h_imageB, *d_imageA, *d_imageB;

    error = cudaMalloc((void**)&d_imageA, size);
    if (error != cudaSuccess) {
        printf("Error.... d_imageA \n");
        return -1;
    }

    h_imageA = image.data;

    clock_t startGPU, endGPU, startCPU, endCPU;
    
    startGPU = clock();
    error = cudaMemcpy(d_imageA, h_imageA, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error... h_imageA a d_imageA \n");
        return -1;
    }


    h_imageB = (uchar*)malloc(sizeGray);
    error = cudaMalloc((void**)&d_imageB, sizeGray);
    if (error != cudaSuccess) {
        printf("Error.... d_imageB \n");
        return -1;
    }
    
    int blockSize = 32;
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);

    grayImageDevice<<<dimGrid, dimBlock>>>(d_imageA, width, height, d_imageB);
    cudaDeviceSynchronize();

    error = cudaMemcpy(h_imageB, d_imageB, sizeGray, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error... d_imageB a h_imageB \n");
        return -1;
    }
    endGPU = clock();
    
    double timeGPU = ((double)(endGPU - startGPU)) / CLOCKS_PER_SEC;
    printf("El tiempo de ejecucion en GPU paralelo es: %.10f\n", timeGPU);
 
    Mat imageGray;
    imageGray.create(height, width, CV_8UC1);
    imageGray.data = h_imageB;
    
    startCPU = clock();
    Mat imageGrayCV;
    cvtColor(image, imageGrayCV, CV_BGR2GRAY);
    endCPU = clock();
    
    double timeCPU = ((double)(endCPU - startCPU)) / CLOCKS_PER_SEC;
    printf("El tiempo de ejecucion en CPU secuencial es: %.10f\n", timeCPU);
    
    //Mat imageSobel;
    //Sobel(imageGray, imageSobel, CV_32F, 1, 0);
    

    imwrite("ferrari_gray.jpg", imageGray);
    //imwrite("ferrari_sobel.jpg", imageSobel);


    free(h_imageB);
    cudaFree(d_imageA); cudaFree(d_imageB);

    return 0;
}