#include <iostream>
#include <math.h>
__global__
void Dense(int n, int m, float *W, float *b, float *x, float* y)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int jn = j*n;
  float r = b[j];
  for (int i = 0; i < n; i++)
    r = r+W[jn+i]*x[i];
  y[j] = r;
}

int main(void)
{
  int N = 1024;
  int M = 2048;
  int blockSize = 128;
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
       Dense<<<numBlocks, blockSize>>>(N, M, W, b, x, y);

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
