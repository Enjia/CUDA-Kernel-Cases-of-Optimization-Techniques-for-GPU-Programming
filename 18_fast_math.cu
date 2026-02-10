/*
注意事项：
- 精度可能下降；需验证数值影响
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define FAST_EXPF(x) expf(x)
#else
#define FAST_EXPF(x) __expf(x)
#endif

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void expf_kernel(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=expf(in[i]);
}


// 优化点：使用近似或硬件加速函数提升速度
__global__ void fast_exp_kernel(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=FAST_EXPF(in[i]);
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h=(float*)malloc(bytes), *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  for(int i=0;i<n;i++) h[i]=((i%100)-50)/50.f;
  float *d_in,*d_o1,*d_o2; CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_o1,bytes)); CHECK(cudaMalloc(&d_o2,bytes));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice));
  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); expf_kernel<<<grid,block>>>(d_in,d_o1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); fast_exp_kernel<<<grid,block>>>(d_in,d_o2,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));
  CHECK(cudaMemcpy(h1,d_o1,bytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("expf: %.3f ms, __expf: %.3f ms\n", t1,t2);

  free(h); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2));
  return 0;
}
