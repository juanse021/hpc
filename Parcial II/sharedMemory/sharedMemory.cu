#include <cuda.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <malloc.h>
#include <opencv2/opencv.hpp>

#define TILE_SIZE 32
#define maskWidth 3;

typedef unsigned char uchar;

using namespace cv;

__global__ void grayImageDevice(const uchar *imgInput, const int width, const int height, uchar *imgOutput) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int index = row * width + col;
        imgOutput[index] = imgInput[(row * width + col) * 3 + 2] * 0.3 + \
                           imgInput[(row * width + col) * 3 + 1] * 0.59 + \
                           imgInput[(row * width + col) * 3] * 0.11;
    }
}

__device__ uchar clamp(double value) {
    if (value > 255)
        value = 255;
    else if (value < 0)
        value = 0;

    return (uchar)value;
}

__global__ void sobelFilter(const uchar *imgInput, const int width, const int height, uchar *imgOutput) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const char sobel_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const char sobel_y[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    __shared__ sMat[TILE_SIZE + maskWidth - 1][TILE_SIZE + maskWidth - 1];

    uint 

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

    uchar *h_imageA, *h_imageB, *h_imageC, *d_imageA, *d_imageB, *d_imageC;

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

    h_imageC = (uchar*)malloc(sizeGray);
    error = cudaMalloc((void**)&d_imageC, sizeGray);
    if (error != cudaSuccess) {
        printf("Error.... d_imageC \n");
        return -1;
    }


    int blockSize = 32;
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);

    grayImageDevice<<<dimGrid, dimBlock>>>(d_imageA, width, height, d_imageB);
    cudaDeviceSynchronize();

    sobelFilter<<<dimGrid, dimBlock>>>(d_imageB, width, height, d_imageC);
    cudaDeviceSynchronize();

    error = cudaMemcpy(h_imageB, d_imageB, sizeGray, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error... d_imageB a h_imageB \n");
        return -1;
    }

    error = cudaMemcpy(h_imageC, d_imageC, sizeGray, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error... d_imageC a h_imageC \n");
        return -1;
    }

    endGPU = clock();
    
    double timeGPU = ((double)(endGPU - startGPU)) / CLOCKS_PER_SEC;
    printf("El tiempo de ejecucion en GPU es: %.10f\n", timeGPU);
 
    Mat imageGray, sobelImage;
    imageGray.create(height, width, CV_8UC1);
    sobelImage.create(height, width, CV_8UC1);
    imageGray.data = h_imageB;
    sobelImage.data = h_imageC;
    
    
    startCPU = clock();
    Mat imageGrayCV, imageSobel_x, abs_imageSobel_x;
    cvtColor(image, imageGrayCV, CV_BGR2GRAY);
    Sobel(imageGrayCV, imageSobel_x, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(imageSobel_x, abs_imageSobel_x);
    endCPU = clock();

    double timeCPU = ((double)(endCPU - startCPU)) / CLOCKS_PER_SEC;
    printf("El tiempo de ejecucion en CPU secuencial es: %.10f\n", timeCPU);
    
    double acceleration = (timeCPU/timeGPU);
    printf("La aceleracion obtenida es de: %.10fX\n", acceleration);    
    

    imwrite("ferrari_gray.jpg", imageGray);
    imwrite("ferrari_sobel.jpg", sobelImage);

    free(h_imageB); free(h_imageC);
    cudaFree(d_imageA); cudaFree(d_imageB); cudaFree(d_imageC);

    return 0;
}
