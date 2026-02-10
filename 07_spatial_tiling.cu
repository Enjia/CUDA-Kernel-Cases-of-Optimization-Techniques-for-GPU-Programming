/*
注意事项：
- 块大小与边界处理需谨慎
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void transpose_naive(const float* in, float* out, int n){
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x<n && y<n) out[x*n+y]=in[y*n+x];
}


// 优化点：按空间块划分提高缓存/共享内存复用
__global__ void transpose_tile(const float* in, float* out, int n){
  __shared__ float tile[32][33];
  int x=blockIdx.x*32+threadIdx.x;
  int y=blockIdx.y*32+threadIdx.y;
  if(x<n && y<n) tile[threadIdx.y][threadIdx.x]=in[y*n+x];
  __syncthreads();
  int tx=blockIdx.y*32+threadIdx.x;
  int ty=blockIdx.x*32+threadIdx.y;
  if(tx<n && ty<n) out[ty*n+tx]=tile[threadIdx.x][threadIdx.y];
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1024; size_t bytes=n*n*sizeof(float);
  float *h=(float*)malloc(bytes), *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  for(int i=0;i<n*n;i++) h[i]=sinf(i*0.01f);
  float *d_in,*d_o1,*d_o2; CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_o1,bytes)); CHECK(cudaMalloc(&d_o2,bytes));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice));
  dim3 block(32,32), grid((n+31)/32,(n+31)/32);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));

  CHECK(cudaEventRecord(s)); transpose_naive<<<grid,block>>>(d_in,d_o1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); transpose_tile<<<grid,block>>>(d_in,d_o2,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));

  CHECK(cudaMemcpy(h1,d_o1,bytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n*n));
  printf("naive: %.3f ms, tile: %.3f ms\n", t1,t2);

  free(h); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2));
  return 0;
}
