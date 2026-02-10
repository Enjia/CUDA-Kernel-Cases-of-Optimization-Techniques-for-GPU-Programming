/*
注意事项：
- 不同架构最优值不同，需实测
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

// 优化点：选择合适block大小平衡占用率与资源
__global__ void saxpy(const float* a, const float* b, float* c, float s, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+s*b[i];
}

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
static int get_opt_block_size(){
  unsigned int minGrid=0, optBlock=0;
  CHECK(hipOccupancyMaxPotentialBlockSize(&minGrid, &optBlock, saxpy, 0, 0));
  return (int)optBlock;
}
#else
static int get_opt_block_size(){
  int minGrid=0, optBlock=0;
  CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &optBlock, saxpy, 0, 0));
  return optBlock;
}
#endif

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h_a=(float*)malloc(bytes), *h_b=(float*)malloc(bytes), *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  for(int i=0;i<n;i++){ h_a[i]=sinf(i*0.01f); h_b[i]=cosf(i*0.01f); }
  float *d_a,*d_b,*d_c1,*d_c2; CHECK(cudaMalloc(&d_a,bytes)); CHECK(cudaMalloc(&d_b,bytes)); CHECK(cudaMalloc(&d_c1,bytes)); CHECK(cudaMalloc(&d_c2,bytes));
  CHECK(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));

  int optBlock = get_opt_block_size();
  dim3 block1(128), grid1((n+block1.x-1)/block1.x);
  dim3 block2(optBlock), grid2((n+block2.x-1)/block2.x);

  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); saxpy<<<grid1,block1>>>(d_a,d_b,d_c1,0.5f,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); saxpy<<<grid2,block2>>>(d_a,d_b,d_c2,0.5f,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));

  CHECK(cudaMemcpy(h1,d_c1,bytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_c2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("block128: %.3f ms, occupancy block %d: %.3f ms\n", t1,optBlock,t2);

  free(h_a); free(h_b); free(h1); free(h2);
  CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c1)); CHECK(cudaFree(d_c2));
  return 0;
}
