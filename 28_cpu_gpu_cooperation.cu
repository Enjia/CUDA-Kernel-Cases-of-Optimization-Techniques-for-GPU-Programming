/*
注意事项：
- 不当划分会使通信开销大于收益
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

// 优化点：合理划分CPU与GPU任务以发挥各自优势
__global__ void compute_gpu(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i]*in[i]+1.f;
}

void compute_cpu(const float* in, float* out, int n){
  for(int i=0;i<n;i++) out[i]=in[i]*in[i]+1.f;
}

float max_diff(const float* a, const float* b, int n){ float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h=(float*)malloc(bytes), *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  for(int i=0;i<n;i++) h[i]=sinf(i*0.01f);
  float *d_in,*d_out; CHECK(cudaMalloc(&d_in,bytes)); CHECK(cudaMalloc(&d_out,bytes));
  CHECK(cudaMemcpy(d_in,h,bytes,cudaMemcpyHostToDevice));
  dim3 block(256), grid((n+block.x-1)/block.x);

  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); compute_gpu<<<grid,block>>>(d_in,d_out,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaMemcpy(h1,d_out,bytes,cudaMemcpyDeviceToHost));

  int half=n/2;
  clock_t c0 = clock();
  compute_cpu(h,h2,half);
  CHECK(cudaMemcpy(d_in,h+half,half*sizeof(float),cudaMemcpyHostToDevice));
  compute_gpu<<<grid,block>>>(d_in,d_out,half);
  CHECK(cudaMemcpy(h2+half,d_out,half*sizeof(float),cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
  clock_t c1 = clock();
  double t2 = 1000.0 * (double)(c1 - c0) / CLOCKS_PER_SEC;

  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("GPU only: %.3f ms, CPU+GPU: %.3f ms\n", t1,t2);

  free(h); free(h1); free(h2);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
  return 0;
}
