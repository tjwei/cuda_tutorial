#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void Dense(int n, int m, float *W, float *b, float *x, float* y)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x*32+threadIdx.y;
  int i0 = threadIdx.y;
  int stride = blockDim.x*32;
  int jn = j*n;
  float r = 0;
  __shared__ float sx[2048];
  __shared__ float temp_out[128];
  for(int i=tid; i<n; i+=stride)
  	sx[i]=x[i];
  __syncthreads();
  for (int i = i0 ; i < n; i+=32)
    r = r+W[jn+i]*sx[i];
  temp_out[i0] = r;
  __syncthreads();
  if (i0 == 0){
  r=b[j];
  for(int i=0;i<32;i++)
        r += temp_out[i];
  y[j] = r;
 }
}

int main(void)
{
  int N = 1024;
  int M = 2048;
  int blockSize = 32;
  dim3 dimBlock(blockSize, 32);
  int numBlocks = 2048/blockSize;
  float *x, *y, *b, *W;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, M*sizeof(float));
  cudaMallocManaged(&b, M*sizeof(float));
  cudaMallocManaged(&W, M*N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f*i;
  }
  for (int j = 0; j < M; j++) {
    b[j] = -0.5f*N*(N-1)+3.5f;
  }
  for (int j=0;j<M;j++)
     for(int i=0;i<N;i++) {
         W[j*N+i]=1;
}
  for(int t=0;t<1000;t++)
       Dense<<<numBlocks, dimBlock>>>(N, M, W, b, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int j = 0; j < M; j++)
    maxError = fmax(maxError, fabs(y[j]-3.5f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(W);
  cudaFree(b);
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
