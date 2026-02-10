/*
注意事项：
- 需数据对齐；过度向量化会限制灵活性
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void add_scalar(const float* a, const float* b, float* c, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i];
}


// 优化点：使用向量类型/指令一次处理多元素
__global__ void add_vec4(const float4* a, const float4* b, float4* c, int n4){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n4){ float4 va=a[i], vb=b[i]; c[i]=make_float4(va.x+vb.x,va.y+vb.y,va.z+vb.z,va.w+vb.w); }
}

__global__ void add_tail(const float* a, const float* b, float* c, int start, int n){
  int i=start+blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i];
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; int n4=n/4; int start=n4*4; int rem=n-start; size_t bytes=n*sizeof(float);
  float *h_a=(float*)malloc(bytes), *h_b=(float*)malloc(bytes), *h1=(float*)malloc(bytes), *h2=(float*)malloc(bytes);
  for(int i=0;i<n;i++){ h_a[i]=sinf(i*0.01f); h_b[i]=cosf(i*0.01f); }
  float *d_a,*d_b,*d_o1,*d_o2;
  CHECK(cudaMalloc(&d_a,bytes)); CHECK(cudaMalloc(&d_b,bytes)); CHECK(cudaMalloc(&d_o1,bytes)); CHECK(cudaMalloc(&d_o2,bytes));
  CHECK(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));

  dim3 block(256), grid((n+block.x-1)/block.x), grid4((n4+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); add_scalar<<<grid,block>>>(d_a,d_b,d_o1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); add_vec4<<<grid4,block>>>((float4*)d_a,(float4*)d_b,(float4*)d_o2,n4); if(rem>0){ dim3 gridt((rem+block.x-1)/block.x); add_tail<<<gridt,block>>>(d_a,d_b,d_o2,start,n);} CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));

  CHECK(cudaMemcpy(h1,d_o1,bytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,bytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("scalar: %.3f ms, float4: %.3f ms\n", t1,t2);

  free(h_a); free(h_b); free(h1); free(h2);
  CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2));
  return 0;
}
