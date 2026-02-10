/*
注意事项：
- 寄存器过多会降低占用率
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void stencil1(const float* in, float* out, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=n) return;
  float c = in[i];
  float l = (i>0)?in[i-1]:0.f;
  float r = (i+1<n)?in[i+1]:0.f;
  out[i]=0.25f*l + 0.5f*c + 0.25f*r;
}


// 优化点：将计算块缓存到寄存器提高复用并降低内存访问
__global__ void stencil2(const float* in, float* out, int n){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int i = tid*2;
  if(i>=n) return;
  float i0 = in[i];
  float i1 = (i+1<n)?in[i+1]:0.f;
  float l0 = (i>0)?in[i-1]:0.f;
  float r1 = (i+2<n)?in[i+2]:0.f;
  out[i]=0.25f*l0 + 0.5f*i0 + 0.25f*i1;
  if(i+1<n) out[i+1]=0.25f*i0 + 0.5f*i1 + 0.25f*r1;
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n = 1<<20; size_t bytes = n*sizeof(float);
  float *h=(float*)malloc(bytes);
  for(int i=0;i<n;i++) h[i]=sinf(i*0.01f);
  float *d_in,*d_o1,*d_o2; CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_o1,bytes)); CHECK(cudaMalloc(&d_o2,bytes));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice));
  dim3 block(256), grid((n+block.x-1)/block.x);
  int n2 = (n+1)/2;
  dim3 grid2((n2+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));

  CHECK(cudaEventRecord(s));
  stencil1<<<grid,block>>>(d_in,d_o1,n);
  CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));

  CHECK(cudaEventRecord(s));
  stencil2<<<grid2,block>>>(d_in,d_o2,n);
  CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));

  float *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  CHECK(cudaMemcpy(h1,d_o1,bytes,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h2,d_o2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("1 output/thread: %.3f ms, 2 outputs/thread: %.3f ms\n", t1,t2);

  free(h); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2));
  return 0;
}
