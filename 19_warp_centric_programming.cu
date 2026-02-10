/*
注意事项：
- 依赖warp语义可能降低可移植性
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void per_thread(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i]*2.f;
}


// 优化点：以warp为协作单位提高通信与同步效率
__global__ void warp_segment(const float* in, float* out, int n){
  int warp = (blockIdx.x*blockDim.x + threadIdx.x)/32;
  int lane = threadIdx.x & 31;
  int base = warp*32;
  int i = base + lane;
  if(i<n) out[i]=in[i]*2.f;
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h=(float*)malloc(bytes), *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  for(int i=0;i<n;i++) h[i]=sinf(i*0.01f);
  float *d_in,*d_o1,*d_o2; CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_o1,bytes)); CHECK(cudaMalloc(&d_o2,bytes));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice));
  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); per_thread<<<grid,block>>>(d_in,d_o1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); warp_segment<<<grid,block>>>(d_in,d_o2,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));
  CHECK(cudaMemcpy(h1,d_o1,bytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("per-thread: %.3f ms, warp-centric: %.3f ms\n", t1,t2);

  free(h); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2));
  return 0;
}
