#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;

// Soal UTS no 1
// Kevin Philips Tanamas - 220711789
// Yosua Budianto - 220711791

#define WIDTH 1024
#define HEIGHT 1024
#define THREADS_PER_BLOCK 256

// Kernel CUDA untuk mengonversi matriks 2D ke array 1D
__global__ void matrixTo1DArray(unsigned char* d_matrix, unsigned char* d_array, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        d_array[idx] = d_matrix[idx];
    }
}

// Fungsi serial untuk konversi matriks ke array 1D
void matrixTo1DArraySerial(unsigned char* matrix, unsigned char* array, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            array[i * width + j] = matrix[i * width + j];
        }
    }
}

// Fungsi untuk membaca file gambar ke dalam Mat menggunakan imdecode
Mat readImageFromPath(const string& path, int flags) {
    ifstream file(path, ios::binary);
    if (!file) {
        cerr << "Error: Tidak dapat membuka file gambar di path: " << path << endl;
        return Mat();
    }

    file.seekg(0, ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, ios::beg);

    vector<unsigned char> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return imdecode(buffer, flags);
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    string path = "D:/Downloads";
    vector<string> myList = { "output1.jpg" };
    vector<Mat> images;

    for (const auto& fileName : myList) {
        Mat img = readImageFromPath(path + "/" + fileName, IMREAD_GRAYSCALE);
        if (!img.empty()) {
            if (img.rows != HEIGHT || img.cols != WIDTH) {
                resize(img, img, Size(WIDTH, HEIGHT));
            }
            images.push_back(img);
        }
        else {
            cerr << "Gagal memuat gambar: " << fileName << endl;
        }
    }

    if (images.empty()) {
        cerr << "Tidak ada gambar yang berhasil dimuat.\n";
        return -1;
    }

    Mat image = images[0];
    unsigned char* h_matrix = image.data;
    unsigned char* h_array = (unsigned char*)malloc(WIDTH * HEIGHT * sizeof(unsigned char));

    // Mengecek apakah alokasi memori berhasil
    if (!h_array) {
        cerr << "Error: Gagal mengalokasikan memori untuk h_array di host." << endl;
        return -1;
    }

    matrixTo1DArraySerial(h_matrix, h_array, WIDTH, HEIGHT);

    unsigned char* d_matrix, * d_array;
    if (cudaMalloc((void**)&d_matrix, WIDTH * HEIGHT * sizeof(unsigned char)) != cudaSuccess) {
        cerr << "Error: Alokasi memori pada device gagal." << endl;
        free(h_array);
        return -1;
    }
    if (cudaMalloc((void**)&d_array, WIDTH * HEIGHT * sizeof(unsigned char)) != cudaSuccess) {
        cerr << "Error: Alokasi memori untuk d_array gagal." << endl;
        cudaFree(d_matrix);
        free(h_array);
        return -1;
    }

    // Memastikan transfer data berhasil
    if (cudaMemcpy(d_matrix, h_matrix, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess) {
        cerr << "Error: Transfer data ke device gagal." << endl;
        cudaFree(d_matrix);
        cudaFree(d_array);
        free(h_array);
        return -1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixTo1DArray << <blocksPerGrid, threadsPerBlock >> > (d_matrix, d_array, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    system("cls");

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (cudaMemcpy(h_array, d_array, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cerr << "Error: Transfer data dari device ke host gagal." << endl;
    }

    // Output sebagian hasil konversi untuk memeriksa keberhasilan
    printf("Hasil konversi matriks 2D ke array 1D:\n");
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // Membersihkan memori
    cudaFree(d_matrix);
    cudaFree(d_array);
    free(h_array);

    printf("Waktu eksekusi kernel GPU: %.3f ms\n", milliseconds);
    printf("Konversi selesai.\n");

    return 0;
}
