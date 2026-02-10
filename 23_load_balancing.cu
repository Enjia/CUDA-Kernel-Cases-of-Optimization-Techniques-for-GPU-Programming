/*
注意事项：
- 动态分配可能引入同步或原子开销
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void static_assign(const int* work, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float sum=0.f; for(int k=0;k<work[i];k++) sum+=sinf(k*0.001f); out[i]=sum;
}


// 优化点：使线程/块工作量均衡减少尾部效应
__global__ void dynamic_assign(const int* work, float* out, int n, int* counter){
  while(true){
    int i=atomicAdd(counter,1); if(i>=n) return;
    float sum=0.f; for(int k=0;k<work[i];k++) sum+=sinf(k*0.001f); out[i]=sum;
  }
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<18; size_t bytes=n*sizeof(float);
  int *h_work=(int*)malloc(n*sizeof(int));
  for(int i=0;i<n;i++) h_work[i]= (i%100)+1;
  int *d_work; float *d_o1,*d_o2; int *d_counter;
  CHECK(cudaMalloc(&d_work,n*sizeof(int))); CHECK(cudaMalloc(&d_o1,bytes)); CHECK(cudaMalloc(&d_o2,bytes)); CHECK(cudaMalloc(&d_counter,sizeof(int)));
  CHECK(cudaMemcpy(d_work,h_work,n*sizeof(int),cudaMemcpyHostToDevice));

  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); static_assign<<<grid,block>>>(d_work,d_o1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaMemset(d_counter,0,sizeof(int)));
  CHECK(cudaEventRecord(s)); dynamic_assign<<<grid,block>>>(d_work,d_o2,n,d_counter); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));

  float *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  CHECK(cudaMemcpy(h1,d_o1,bytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("static: %.3f ms, dynamic: %.3f ms\n", t1,t2);

  free(h_work); free(h1); free(h2);
  CHECK(cudaFree(d_work)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2)); CHECK(cudaFree(d_counter));
  return 0;
}
