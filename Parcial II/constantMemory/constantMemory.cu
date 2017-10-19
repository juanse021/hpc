#include <cuda.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <libgen.h>
#include <malloc.h>
#include <opencv2/opencv.hpp>

#define maskWidth 3

__constant__ char MX[maskWidth * maskWidth];
__constant__ char MY[maskWidth * maskWidth];

typedef unsigned char uchar;

using namespace cv;

__global__ void grayImageDevice(const uchar *imgInput, const int width, const int height, uchar *imgOutput) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int index = row * width + col;
        imgOutput[index] = imgInput[index * 3 + 2] * 0.3 + \
                           imgInput[index * 3 + 1] * 0.59 + \
                           imgInput[index * 3] * 0.11;
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
    
    double magnitude_x = 0;
    double magnitude_y = 0;

    int starti = row - (maskWidth/2);
    int startj = col - (maskWidth/2);

    if (col > width && row > height)
        return;

    for (int i = 0; i < maskWidth; i++) {
        int focus_x = starti + i;
        for (int j = 0; j < maskWidth; j++) {
            int focus_y = startj + j;
            if (focus_x >= 0 && focus_x < height && focus_y >= 0 && focus_y < width) {
                magnitude_x += imgInput[focus_y + focus_x * width] * MX[i * maskWidth + j];
                magnitude_y += imgInput[focus_y + focus_x * width] * MY[i * maskWidth + j];
            }
        }
    }

    double magnitude = sqrt(magnitude_x * magnitude_x + magnitude_y * magnitude_y);
    imgOutput[col + row * width] = clamp(magnitude);

}

void wrTimes(Size s, char *imgname, double time) {
    long sz = s.width * s.height;
    FILE *f = fopen("times/data.txt", "a");
    if (f == NULL)
        printf("Error opening file\n");
    else
        fprintf(f, "%ld %s %lf\n", sz, imgname, time);

    fclose(f);
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

    const char h_sobel_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const char h_sobel_y[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

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

    error = cudaMemcpyToSymbol(MX, h_sobel_x, sizeof(char) * maskWidth * maskWidth);
    if (error != cudaSuccess) {
        printf("Error.... MX \n");
        return -1;
    }

    error = cudaMemcpyToSymbol(MY, h_sobel_y, sizeof(char) * maskWidth * maskWidth);
    if (error != cudaSuccess) {
        printf("Error.... MY \n");
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
    
    wrTimes(s, strtok(basename(imageName), "."), timeGPU);
 
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
    

    //imwrite(strcat(strtok(basename(imageName), "."), "_gray.jpg"), imageGray);
    //imwrite(strcat(strtok(basename(imageName), "."), "_sobel.jpg"), sobelImage);    


    free(h_imageB); free(h_imageC);
    cudaFree(MX); cudaFree(MY); cudaFree(d_imageA); cudaFree(d_imageB); cudaFree(d_imageC);

    return 0;
}
