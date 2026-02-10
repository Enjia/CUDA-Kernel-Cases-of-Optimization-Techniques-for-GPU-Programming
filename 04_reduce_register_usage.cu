/*
注意事项：
- 寄存器不足会溢出到本地内存
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void vec_add_heavy(const float* a, const float* b, float* c, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=n) return;
  float t0=a[i]; float t1=b[i];
  float t2=t0+t1; float t3=t2*1.1f; float t4=t3-0.1f; float t5=t4*0.9f;
  c[i]=t5;
}


// 优化点：精简临时变量与数据类型以减轻寄存器压力
__global__ void vec_add_light(const float* a, const float* b, float* c, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=n) return;
  float v = (a[i]+b[i])*1.1f;
  c[i]= (v-0.1f)*0.9f;
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(float);
  float *h_a=(float*)malloc(bytes), *h_b=(float*)malloc(bytes);
  for(int i=0;i<n;i++){ h_a[i]=sinf(i); h_b[i]=cosf(i); }
  float *d_a,*d_b,*d_c1,*d_c2; CHECK(cudaMalloc(&d_a,bytes)); CHECK(cudaMalloc(&d_b,bytes)); CHECK(cudaMalloc(&d_c1,bytes)); CHECK(cudaMalloc(&d_c2,bytes));
  CHECK(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));

  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); vec_add_heavy<<<grid,block>>>(d_a,d_b,d_c1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); vec_add_light<<<grid,block>>>(d_a,d_b,d_c2,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));

  float *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  CHECK(cudaMemcpy(h1,d_c1,bytes,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h2,d_c2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("heavy temps: %.3f ms, light temps: %.3f ms\n", t1,t2);

  free(h_a); free(h_b); free(h1); free(h2);
  CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c1)); CHECK(cudaFree(d_c2));
  return 0;
}
