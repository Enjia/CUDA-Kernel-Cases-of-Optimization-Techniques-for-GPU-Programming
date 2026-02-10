/*
注意事项：
- 容量有限；共享内存需避免bank冲突；常量内存适合只读广播
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

const int R = 4;
__constant__ float c_k[2*R+1];

__global__ void conv_global(const float* in, float* out, const float* k, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=n) return;
  float sum = 0.f;
  for(int r=-R;r<=R;r++){
    int idx = i + r;
    if(idx>=0 && idx<n) sum += in[idx]*k[r+R];
  }
  out[i]=sum;
}


// 优化点：将热点数据放入共享/常量/纹理内存以降低访问延迟
__global__ void conv_const(const float* in, float* out, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=n) return;
  float sum = 0.f;
  for(int r=-R;r<=R;r++){
    int idx = i + r;
    if(idx>=0 && idx<n) sum += in[idx]*c_k[r+R];
  }
  out[i]=sum;
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f;
  for(int i=0;i<n;i++){
    float d=fabsf(a[i]-b[i]);
    if(d>m) m=d;
  }
  return m;
}

int main(){
  int n = 1<<20;
  size_t bytes = n*sizeof(float);
  float *h_in=(float*)malloc(bytes);
  float *h_out1=(float*)malloc(bytes);
  float *h_out2=(float*)malloc(bytes);
  float h_k[2*R+1];
  for(int i=0;i<n;i++) h_in[i]=sinf(i*0.001f);
  for(int i=0;i<2*R+1;i++) h_k[i]=0.1f*(i+1);

  float *d_in,*d_out1,*d_out2,*d_k;
  CHECK(cudaMalloc(&d_in,bytes));
  CHECK(cudaMalloc(&d_out1,bytes));
  CHECK(cudaMalloc(&d_out2,bytes));
  CHECK(cudaMalloc(&d_k,(2*R+1)*sizeof(float)));
  CHECK(cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_k,h_k,(2*R+1)*sizeof(float),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(c_k,h_k,(2*R+1)*sizeof(float)));

  dim3 block(256); dim3 grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));

  CHECK(cudaEventRecord(s));
  conv_global<<<grid,block>>>(d_in,d_out1,d_k,n);
  CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));

  CHECK(cudaEventRecord(s));
  conv_const<<<grid,block>>>(d_in,d_out2,n);
  CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));

  CHECK(cudaMemcpy(h_out1,d_out1,bytes,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_out2,d_out2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h_out1,h_out2,n));
  printf("global kernel: %.3f ms, constant kernel: %.3f ms\n", t1,t2);

  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out1)); CHECK(cudaFree(d_out2)); CHECK(cudaFree(d_k));
  free(h_in); free(h_out1); free(h_out2);
  return 0;
}
