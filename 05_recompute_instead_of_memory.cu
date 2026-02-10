/*
注意事项：
- 计算代价和精度需评估
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void use_index_array(const float* in, const int* idx, float* out, int n){
  int i=blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=n) return;
  int j=idx[i];
  out[i]=in[j]*1.001f;
}


// 优化点：用计算替代内存读写以隐藏访问延迟
__global__ void recompute_index(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=n) return;
  int j=(i*7) % n;
  out[i]=in[j]*1.001f;
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h=(float*)malloc(bytes); int *h_idx=(int*)malloc(n*sizeof(int));
  for(int i=0;i<n;i++){ h[i]=sinf(i*0.01f); h_idx[i]=(i*7)%n; }
  float *d_in,*d_o1,*d_o2; int *d_idx; CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_o1,bytes)); CHECK(cudaMalloc(&d_o2,bytes)); CHECK(cudaMalloc(&d_idx,n*sizeof(int)));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_idx,h_idx,n*sizeof(int),cudaMemcpyHostToDevice));

  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); use_index_array<<<grid,block>>>(d_in,d_idx,d_o1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); recompute_index<<<grid,block>>>(d_in,d_o2,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));

  float *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  CHECK(cudaMemcpy(h1,d_o1,bytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("index array: %.3f ms, recompute: %.3f ms\n", t1,t2);

  free(h); free(h_idx); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2)); CHECK(cudaFree(d_idx));
  return 0;
}
