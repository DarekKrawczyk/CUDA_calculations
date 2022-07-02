#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>

/*
* Kernel code, adding two matrices together A<-A+B
*/
__global__ void add(float* A, float* B, int M, int N) {

	int i = blockDim.x*blockIdx.x+threadIdx.x;
	int j = blockDim.y*blockIdx.y+threadIdx.y;
	B[i * M + j] += A[i * M + j];
}

void fill(float* tab, float value, int N, int M) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			tab[i * N + j] = value;
		}
	}
}

void show(float* tab, std::string text) {
	std::cout << text << '\n';
	for (int i = 0; i < sizeof(tab); i++) {
		for (int j = 0; j < sizeof(tab); j++) {
			std::cout << tab[i * sizeof(tab) + j] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
}

int main(){
	const int N = 10;
	const int M = 10;
	float* tab = new float[N*M];
	float* tab_ = new float[N*M];
	size_t size = N * M * sizeof(float);
	float* A_GPU;
	float* B_GPU;

	fill(tab, 5.0f, N, M);
	fill(tab_, 2.0f, N, M);

	show(tab,"A-Przed");
	show(tab_,"B-Przed");


	cudaMalloc((void**)&A_GPU, size);
	cudaMalloc((void**)&B_GPU, size);

	cudaMemcpy(A_GPU, tab, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, tab_, size, cudaMemcpyHostToDevice);

	dim3 block(5,5);
	dim3 threats(2,2);
	dim3 shared(threats.x, threats.y);

	add<<<block,threats >>>(A_GPU,B_GPU,M,N);

	cudaMemcpy(tab, B_GPU, size, cudaMemcpyDeviceToHost);

	show(tab,"Po");

	cudaFree(A_GPU);
	cudaFree(B_GPU);
	delete[]tab;
}
