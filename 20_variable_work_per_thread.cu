/*
注意事项：
- 可能导致负载不均
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void fixed_stride(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i]*1.01f;
}


// 优化点：按数据分布调整线程工作量
__global__ void grid_stride(const float* in, float* out, int n){
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=blockDim.x*gridDim.x) out[i]=in[i]*1.01f;
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<24; size_t bytes=n*sizeof(float);
  float *h=(float*)malloc(bytes), *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  for(int i=0;i<n;i++) h[i]=sinf(i*0.001f);
  float *d_in,*d_o1,*d_o2; CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_o1,bytes)); CHECK(cudaMalloc(&d_o2,bytes));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice));
  dim3 block(256), grid(1024);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); fixed_stride<<<(n+block.x-1)/block.x,block>>>(d_in,d_o1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); grid_stride<<<grid,block>>>(d_in,d_o2,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));
  CHECK(cudaMemcpy(h1,d_o1,bytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("fixed: %.3f ms, grid-stride: %.3f ms\n", t1,t2);

  free(h); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2));
  return 0;
}
