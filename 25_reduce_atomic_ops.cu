/*
注意事项：
- 替代方案可能增加共享内存与同步
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

const int BINS=256;

__global__ void hist_global(const unsigned char* in, int* hist, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) atomicAdd(&hist[in[i]],1);
}


// 优化点：用分块/warp归约减少原子争用
__global__ void hist_block(const unsigned char* in, int* hist, int n){
  __shared__ int local[BINS];
  for(int i=threadIdx.x;i<BINS;i+=blockDim.x) local[i]=0; __syncthreads();
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) atomicAdd(&local[in[i]],1);
  __syncthreads();
  for(int i=threadIdx.x;i<BINS;i+=blockDim.x) atomicAdd(&hist[i], local[i]);
}

int max_diff_i(const int* a, const int* b, int n){ int m=0; for(int i=0;i<n;i++){int d=abs(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(unsigned char);
  unsigned char *h=(unsigned char*)malloc(bytes); for(int i=0;i<n;i++) h[i]=(unsigned char)(i%256);
  unsigned char *d_in; int *d_h1,*d_h2; CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_h1,BINS*sizeof(int))); CHECK(cudaMalloc(&d_h2,BINS*sizeof(int)));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_h1,0,BINS*sizeof(int))); CHECK(cudaMemset(d_h2,0,BINS*sizeof(int)));
  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); hist_global<<<grid,block>>>(d_in,d_h1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); hist_block<<<grid,block>>>(d_in,d_h2,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));

  int *h1=(int*)malloc(BINS*sizeof(int)), *h2=(int*)malloc(BINS*sizeof(int));
  CHECK(cudaMemcpy(h1,d_h1,BINS*sizeof(int),cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_h2,BINS*sizeof(int),cudaMemcpyDeviceToHost));
  printf("max diff: %d\n", max_diff_i(h1,h2,BINS));
  printf("global atomics: %.3f ms, block local: %.3f ms\n", t1,t2);

  free(h); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_h1)); CHECK(cudaFree(d_h2));
  return 0;
}
