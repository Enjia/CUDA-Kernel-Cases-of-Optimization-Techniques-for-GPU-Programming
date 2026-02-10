/*
注意事项：
- 需平衡预计算时间与存储开销
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

const int LUT = 1024;
__constant__ float c_lut[LUT];

__global__ void compute_sin(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=sinf(in[i]);
}


// 优化点：预先计算并缓存重复使用结果
__global__ void compute_lut(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){
    int idx = (int)(in[i]* (LUT-1)); if(idx<0) idx=0; if(idx>=LUT) idx=LUT-1; out[i]=c_lut[idx];
  }
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h=(float*)malloc(bytes), *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  float h_lut[LUT];
  for(int i=0;i<n;i++) h[i]=fmodf(i*0.001f,1.f);
  for(int i=0;i<LUT;i++) h_lut[i]=sinf(i/(float)(LUT-1));
  CHECK(cudaMemcpyToSymbol(c_lut,h_lut,LUT*sizeof(float)));

  float *d_in,*d_o1,*d_o2; CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_o1,bytes)); CHECK(cudaMalloc(&d_o2,bytes));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice));
  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); compute_sin<<<grid,block>>>(d_in,d_o1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); compute_lut<<<grid,block>>>(d_in,d_o2,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));
  CHECK(cudaMemcpy(h1,d_o1,bytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("sinf: %.3f ms, LUT: %.3f ms\n", t1,t2);

  free(h); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2));
  return 0;
}
