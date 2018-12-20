/*  Shubham
 *  Gulati
 *  sgulati3
 */

#ifndef A3_HPP
#define A3_HPP
#define PI 3.141592654f
#include <cuda_runtime_api.h>
#include <vector>
#include <algorithm>
#include <functional>

__device__
float kernelSummation(float x) {
    return (1/(pow((2*PI),1/2)))*exp(-x*x/2);
}

__global__
void gauss(int n, float h, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float S = 0.0;
    if (i < n) {
        for (int j=1; j<=n; j++) {
            S+= kernelSummation((x[i] - x[j])/h);
        }
    }
    
    y[i] = S/(n*h);
}

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
    //implementing cuda code here
    int size = n* sizeof(float);

    float* cuda_x;
    float* cuda_y;
    cudaMalloc(&cuda_x, size);
    cudaMalloc(&cuda_y, size);

    cudaMemcpy(cuda_x, x.data(), size, cudaMemcpyHostToDevice);

    const int block_size = 1024;
    int num_blocks = (n + block_size - 1)/block_size;

    gauss<<<num_blocks, block_size>>>(n, h, cuda_x, cuda_y);
    cudaMemcpy(y.data(), cuda_y, size, cudaMemcpyDeviceToHost);

    cudaFree(cuda_x);
    cudaFree(cuda_y);

} // gaussian_kde


#endif // A3_HPP