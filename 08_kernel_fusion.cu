/*
注意事项：
- 可能增加寄存器/共享内存压力
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void scale_kernel(const float* in, float* out, float s, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i]*s;
}


// 优化点：合并内核减少中间存储和启动开销
__global__ void bias_kernel(const float* in, float* out, float b, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i]+b;
}

__global__ void fused_kernel(const float* in, float* out, float s, float b, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i]*s + b;
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h=(float*)malloc(bytes), *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  for(int i=0;i<n;i++) h[i]=sinf(i*0.01f);
  float *d_in,*d_tmp,*d_o1,*d_o2; CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_tmp,bytes)); CHECK(cudaMalloc(&d_o1,bytes)); CHECK(cudaMalloc(&d_o2,bytes));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice));
  float s=1.5f,b=0.5f;
  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t evs,eve; CHECK(cudaEventCreate(&evs)); CHECK(cudaEventCreate(&eve));

  CHECK(cudaEventRecord(evs));
  scale_kernel<<<grid,block>>>(d_in,d_tmp,s,n); bias_kernel<<<grid,block>>>(d_tmp,d_o1,b,n);
  CHECK(cudaEventRecord(eve)); CHECK(cudaEventSynchronize(eve));
  float t1; CHECK(cudaEventElapsedTime(&t1,evs,eve));

  CHECK(cudaEventRecord(evs)); fused_kernel<<<grid,block>>>(d_in,d_o2,s,b,n); CHECK(cudaEventRecord(eve)); CHECK(cudaEventSynchronize(eve));
  float t2; CHECK(cudaEventElapsedTime(&t2,evs,eve));

  CHECK(cudaMemcpy(h1,d_o1,bytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("2 kernels: %.3f ms, fused: %.3f ms\n", t1,t2);

  free(h); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_tmp)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2));
  return 0;
}
