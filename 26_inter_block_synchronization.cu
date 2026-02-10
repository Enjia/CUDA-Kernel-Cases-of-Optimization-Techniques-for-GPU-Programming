/*
注意事项：
- GPU不支持直接块间同步，需谨慎设计
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define USE_COOP 0
#else
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define USE_COOP 1
#endif

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

#if USE_COOP
__global__ void coop_kernel(float* data, int n){
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int i=tid;i<n;i+=stride) data[i]*=2.f;
  grid.sync();
  for(int i=tid;i<n;i+=stride) data[i]+=1.f;
}
#endif

// 优化点：通过多阶段kernel或全局同步协调block间依赖
__global__ void step1(float* data, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) data[i]*=2.f; }
__global__ void step2(float* data, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) data[i]+=1.f; }

float max_diff(const float* a, const float* b, int n){ float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h=(float*)malloc(bytes), *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  for(int i=0;i<n;i++) h[i]=sinf(i*0.01f);
  float *d1,*d2; CHECK(cudaMalloc(&d1,bytes)); CHECK(cudaMalloc(&d2,bytes));
  CHECK(cudaMemcpy(d1,h,bytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d2,h,bytes,cudaMemcpyHostToDevice));

  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); step1<<<grid,block>>>(d1,n); step2<<<grid,block>>>(d1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));

  float t2=-1.f;
  const char* path = "coop";
#if USE_COOP
  int dev; CHECK(cudaGetDevice(&dev)); cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop,dev));
  int maxBlocksPerSM=0; CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, coop_kernel, block.x, 0));
  int maxCoopBlocks = maxBlocksPerSM * prop.multiProcessorCount;
  if(prop.cooperativeLaunch && maxCoopBlocks > 0){
    dim3 coopGrid((grid.x <= (unsigned)maxCoopBlocks) ? grid.x : maxCoopBlocks);
    void* args[] = { &d2, &n };
    CHECK(cudaEventRecord(s));
    cudaError_t launchErr = cudaLaunchCooperativeKernel((void*)coop_kernel, coopGrid, block, args);
    if(launchErr == cudaSuccess){
      CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
      CHECK(cudaEventElapsedTime(&t2,s,e));
    } else {
      path = "multi-kernel (coop launch failed)";
      CHECK(cudaGetLastError());
      CHECK(cudaEventRecord(s));
      step1<<<grid,block>>>(d2,n); step2<<<grid,block>>>(d2,n);
      CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
      CHECK(cudaEventElapsedTime(&t2,s,e));
    }
  } else {
    path = "multi-kernel (coop unsupported)";
    CHECK(cudaEventRecord(s));
    step1<<<grid,block>>>(d2,n); step2<<<grid,block>>>(d2,n);
    CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
    CHECK(cudaEventElapsedTime(&t2,s,e));
  }
#else
  path = "multi-kernel (no coop)";
  CHECK(cudaEventRecord(s));
  step1<<<grid,block>>>(d2,n); step2<<<grid,block>>>(d2,n);
  CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  CHECK(cudaEventElapsedTime(&t2,s,e));
#endif

  CHECK(cudaMemcpy(h1,d1,bytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("multi-kernel: %.3f ms, coop path: %s, time: %.3f ms\n", t1, path, t2);

  free(h); free(h1); free(h2);
  CHECK(cudaFree(d1)); CHECK(cudaFree(d2));
  return 0;
}
