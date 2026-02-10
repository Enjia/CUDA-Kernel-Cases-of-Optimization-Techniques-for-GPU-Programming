/*
注意事项：
- 增加启动与中间存储开销
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void complex_kernel(const float* a, const float* b, float* c, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float t = a[i]*b[i]; t = t + 1.f; t = t*t; c[i]=sqrtf(t);
}


// 优化点：拆分复杂内核以降低资源压力或提升并行度
__global__ void kernel1(const float* a, const float* b, float* tmp, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float t=a[i]*b[i]; tmp[i]=t+1.f;
}

__global__ void kernel2(const float* tmp, float* c, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float t=tmp[i]; c[i]=sqrtf(t*t);
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h_a=(float*)malloc(bytes), *h_b=(float*)malloc(bytes), *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  for(int i=0;i<n;i++){ h_a[i]=sinf(i*0.01f); h_b[i]=cosf(i*0.01f); }
  float *d_a,*d_b,*d_tmp,*d_o1,*d_o2; CHECK(cudaMalloc(&d_a,bytes)); CHECK(cudaMalloc(&d_b,bytes)); CHECK(cudaMalloc(&d_tmp,bytes)); CHECK(cudaMalloc(&d_o1,bytes)); CHECK(cudaMalloc(&d_o2,bytes));
  CHECK(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));
  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); complex_kernel<<<grid,block>>>(d_a,d_b,d_o1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); kernel1<<<grid,block>>>(d_a,d_b,d_tmp,n); kernel2<<<grid,block>>>(d_tmp,d_o2,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));
  CHECK(cudaMemcpy(h1,d_o1,bytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("single kernel: %.3f ms, split: %.3f ms\n", t1,t2);

  free(h_a); free(h_b); free(h1); free(h2);
  CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_tmp)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2));
  return 0;
}
