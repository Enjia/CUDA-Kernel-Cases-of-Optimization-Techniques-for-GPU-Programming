/*
注意事项：
- 调优成本高；需稳定基准
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

// 优化点：自动搜索参数以获得最佳性能
__global__ void kernel(const float* a, const float* b, float* c, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]*b[i]+1.f;
}

float time_launch(const float* d_a, const float* d_b, float* d_c, int n, int block){
  dim3 blk(block), grd((n+block-1)/block);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); kernel<<<grd,blk>>>(d_a,d_b,d_c,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t; CHECK(cudaEventElapsedTime(&t,s,e)); CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
  return t;
}

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h_a=(float*)malloc(bytes), *h_b=(float*)malloc(bytes);
  for(int i=0;i<n;i++){ h_a[i]=sinf(i*0.01f); h_b[i]=cosf(i*0.01f); }
  float *d_a,*d_b,*d_c; CHECK(cudaMalloc(&d_a,bytes)); CHECK(cudaMalloc(&d_b,bytes)); CHECK(cudaMalloc(&d_c,bytes));
  CHECK(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));

  int candidates[] = {64,128,256,512};
  float best=FLT_MAX; int bestBlock=0;
  for(int i=0;i<4;i++){
    float t=time_launch(d_a,d_b,d_c,n,candidates[i]);
    if(t<best){ best=t; bestBlock=candidates[i]; }
  }
  printf("best block: %d, time: %.3f ms\n", bestBlock, best);

  free(h_a); free(h_b);
  CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c));
  return 0;
}
