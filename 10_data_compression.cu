/*
注意事项：
- 压缩/解压开销与精度需评估
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void compute_float(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i]*0.5f+1.f;
}


// 优化点：降低数据传输与存储成本，提高有效带宽
__global__ void compute_packed(const uint32_t* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  int pack = i/4; int lane = i%4;
  uint32_t v = in[pack];
  uint8_t b = (v >> (lane*8)) & 0xff;
  float f = (float)b * (1.f/255.f);
  out[i]=f*0.5f+1.f;
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t fbytes=n*sizeof(float); int n_pack=(n+3)/4; size_t pbytes=n_pack*sizeof(uint32_t);
  float *h=(float*)malloc(fbytes), *h1=(float*)malloc(fbytes), *h2=(float*)malloc(fbytes);
  uint32_t *hpack=(uint32_t*)malloc(pbytes);
  for(int i=0;i<n;i++) h[i]=fmodf((float)i,255.f)/255.f;
  for(int i=0;i<n_pack;i++){
    uint32_t v=0; for(int j=0;j<4;j++){ int idx=i*4+j; uint8_t b=0; if(idx<n) b=(uint8_t)(h[idx]*255.f); v |= ((uint32_t)b)<<(j*8); } hpack[i]=v;
  }

  float *d_in,*d_o1,*d_o2; uint32_t *d_pack;
  CHECK(cudaMalloc(&d_in,fbytes)); CHECK(cudaMalloc(&d_o1,fbytes)); CHECK(cudaMalloc(&d_o2,fbytes)); CHECK(cudaMalloc(&d_pack,pbytes));
  CHECK(cudaMemcpy(d_in,h,fbytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_pack,hpack,pbytes,cudaMemcpyHostToDevice));

  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); compute_float<<<grid,block>>>(d_in,d_o1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); compute_packed<<<grid,block>>>(d_pack,d_o2,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));

  CHECK(cudaMemcpy(h1,d_o1,fbytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2,d_o2,fbytes,cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h1,h2,n));
  printf("float: %.3f ms, packed8: %.3f ms\n", t1,t2);

  free(h); free(h1); free(h2); free(hpack);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_o1)); CHECK(cudaFree(d_o2)); CHECK(cudaFree(d_pack));
  return 0;
}
