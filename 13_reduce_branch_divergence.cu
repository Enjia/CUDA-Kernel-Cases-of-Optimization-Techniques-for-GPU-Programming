/*
注意事项：
- 可能需要重排数据或算法
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void classify_branch(const float* in, int* out, float thresh, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  if(in[i] > thresh) out[i]=1; else out[i]=0;
}


// 优化点：让warp内线程尽量走相同路径
__global__ void classify_mask(const float* in, int* out, float thresh, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  out[i] = (in[i] > thresh);
}

int max_diff_i(const int* a, const int* b, int n){
  int m=0; for(int i=0;i<n;i++){int d=abs(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h=(float*)malloc(bytes); for(int i=0;i<n;i++) h[i]=sinf(i*0.01f);
  int *h1=(int*)malloc(n*sizeof(int)), *h2=(int*)malloc(n*sizeof(int));
  float *d_in; int *d_o1,*d_o2; CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_o1,n*sizeof(int))); CHECK(cudaMalloc(&d_o2,n*sizeof(int)));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice));
  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); classify_branch<<<grid,block>>>(d_in,d_o1,0.1f,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); classify_mask<<<grid,block>>>(d_in,d_o2,0.1f,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));
  CHECK(cudaMemcpy(h1,d_o1,n*sizeof(int),cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,n*sizeof(int),cudaMemcpyDeviceToHost));
  printf("max diff: %d\n", max_diff_i(h1,h2,n));
  printf("branch: %.3f ms, mask: %.3f ms\n", t1,t2);

  free(h); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2));
  return 0;
}
