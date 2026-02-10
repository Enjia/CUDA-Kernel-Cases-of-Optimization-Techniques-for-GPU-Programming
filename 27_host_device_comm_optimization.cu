/*
注意事项：
- PCIe带宽有限，避免频繁小传输
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

// 优化点：减少数据传输次数，使用异步拷贝与流并行
__global__ void scale(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i]*2.f;
}

float max_diff(const float* a, const float* b, int n){ float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h=(float*)malloc(bytes), *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  for(int i=0;i<n;i++) h[i]=sinf(i*0.01f);

  float *d_in,*d_out; CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_out,bytes));
  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));

  CHECK(cudaEventRecord(s));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice)); scale<<<grid,block>>>(d_in,d_out,n); CHECK(cudaMemcpy(h1,d_out,bytes,cudaMemcpyDeviceToHost));
  CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));

  float *h_pin_in,*h_pin_out; CHECK(cudaMallocHost(&h_pin_in,bytes)); CHECK(cudaMallocHost(&h_pin_out,bytes));
  for(int i=0;i<n;i++) h_pin_in[i]=h[i];
  cudaStream_t st; CHECK(cudaStreamCreate(&st));
  CHECK(cudaEventRecord(s,st));
  CHECK(cudaMemcpyAsync(d_in,h_pin_in,bytes,cudaMemcpyHostToDevice,st)); scale<<<grid,block,0,st>>>(d_in,d_out,n); CHECK(cudaMemcpyAsync(h_pin_out,d_out,bytes,cudaMemcpyDeviceToHost,st));
  CHECK(cudaEventRecord(e,st)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));
  for(int i=0;i<n;i++) h2[i]=h_pin_out[i];

  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("sync copy: %.3f ms, async pinned: %.3f ms\n", t1,t2);

  CHECK(cudaStreamDestroy(st)); CHECK(cudaFreeHost(h_pin_in)); CHECK(cudaFreeHost(h_pin_out));
  free(h); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
  return 0;
}
