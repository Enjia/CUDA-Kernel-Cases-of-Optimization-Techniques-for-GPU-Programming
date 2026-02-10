/*
注意事项：
- 代码体积增大；展开因子需调优
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void reduce_no_unroll(const float* in, float* out, int n){
  __shared__ float s[256];
  int tid=threadIdx.x; int i=blockIdx.x*blockDim.x*2+tid;
  float sum=0.f; if(i<n) sum+=in[i]; if(i+blockDim.x<n) sum+=in[i+blockDim.x];
  s[tid]=sum; __syncthreads();
  for(int stride=blockDim.x/2; stride>0; stride>>=1){ if(tid<stride) s[tid]+=s[tid+stride]; __syncthreads(); }
  if(tid==0) out[blockIdx.x]=s[0];
}


// 优化点：减少分支与循环控制开销，提高指令级并行
__global__ void reduce_unroll(const float* in, float* out, int n){
  __shared__ float s[256];
  int tid=threadIdx.x; int i=blockIdx.x*blockDim.x*2+tid;
  float sum=0.f; if(i<n) sum+=in[i]; if(i+blockDim.x<n) sum+=in[i+blockDim.x];
  s[tid]=sum; __syncthreads();
  #pragma unroll
  for(int stride=blockDim.x/2; stride>0; stride>>=1){ if(tid<stride) s[tid]+=s[tid+stride]; __syncthreads(); }
  if(tid==0) out[blockIdx.x]=s[0];
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h=(float*)malloc(bytes); for(int i=0;i<n;i++) h[i]=1.f;
  float *d_in,*d_o1,*d_o2; int blocks=(n+512-1)/512;
  CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_o1,blocks*sizeof(float))); CHECK(cudaMalloc(&d_o2,blocks*sizeof(float)));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice));
  dim3 block(256), grid(blocks);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); reduce_no_unroll<<<grid,block>>>(d_in,d_o1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); reduce_unroll<<<grid,block>>>(d_in,d_o2,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));
  float *h1=(float*)malloc(blocks*sizeof(float)), *h2=(float*)malloc(blocks*sizeof(float));
  CHECK(cudaMemcpy(h1,d_o1,blocks*sizeof(float),cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,blocks*sizeof(float),cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,blocks));
  printf("no unroll: %.3f ms, unroll: %.3f ms\n", t1,t2);

  free(h); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2));
  return 0;
}
