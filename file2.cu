#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"  // Including the kernel.cu file

using namespace std;
using namespace cv;

// Function to create an image with a diagonal line
Mat createDiagonalImage(int width, int height) {
    Mat img(height, width, CV_32F, Scalar(255));
    for (int i = 0; i < height; ++i) {
        img.at<float>(i, i) = 0; // Draw black line
        if (i < width) {
            img.at<float>(i, width - i - 1) = 0; // Draw black line
        }
    }
    return img;
}

// Function to create an image with a black circle
Mat createCircleImage(int width, int height) {
    Mat img(height, width, CV_32F, Scalar(255));
    circle(img, Point(width / 2, height / 2), 50, Scalar(0), -1); // Radius 50
    return img;
}

// Host function for matrix summation (for validation)
void sumMatrixOnHost(const Mat& A, const Mat& B, Mat& C, const int nx, const int ny) {
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            C.at<float>(y, x) = A.at<float>(y, x) + B.at<float>(y, x);
            if (C.at<float>(y, x) > 255) C.at<float>(y, x) = 255;
        }
    }
}

int main() {
    // Set OpenCV log level to ERROR to hide info messages
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    // Create two simple images of size 512x512
    int width = 512, height = 512; // Image size
    Mat img1 = createDiagonalImage(width, height);
    Mat img2 = createCircleImage(width, height);

    int nx = img1.cols;
    int ny = img1.rows;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // Allocate memory for results
    Mat hostRef(img1.size(), CV_32F);
    Mat gpuRef(img1.size(), CV_32F);

    // Allocate memory on GPU
    float* d_MatA, * d_MatB, * d_MatC;
    if (cudaMalloc((void**)&d_MatA, nBytes) != cudaSuccess ||
        cudaMalloc((void**)&d_MatB, nBytes) != cudaSuccess ||
        cudaMalloc((void**)&d_MatC, nBytes) != cudaSuccess) {
        cerr << "Error: Unable to allocate device memory" << endl;
        return -1;
    }

    // Copy data to device
    cudaMemcpy(d_MatA, img1.ptr<float>(), nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, img2.ptr<float>(), nBytes, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(32, 32);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // Call the CUDA kernel to sum matrices on the GPU
    sumMatrixOnGPU2D << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(error) << endl;
        return -1;
    }

    cudaDeviceSynchronize();

    // Copy result from GPU to host
    cudaMemcpy(gpuRef.ptr<float>(), d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // Save the result as output image
    gpuRef.convertTo(gpuRef, CV_8U); // Convert back to 8-bit
    imwrite("output.jpg", gpuRef); // Save output image

    // Display the result
    imshow("Result of Summation", gpuRef);
    waitKey(0); // Wait for user input

    std::cout << "Press Enter to exit...";
    std::cin.get(); // Wait for user input

    // Free memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);
    return 0;
}
