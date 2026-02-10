/*
注意事项：
- 数据布局需匹配；不对齐或步长访问会降效
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

typedef struct { float x,y,z; } Vec3;

__global__ void add_aos(const Vec3* a, const Vec3* b, Vec3* c, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  Vec3 va=a[i], vb=b[i]; c[i].x=va.x+vb.x; c[i].y=va.y+vb.y; c[i].z=va.z+vb.z;
}


// 优化点：让warp访问连续地址提升带宽利用
__global__ void add_soa(const float* ax,const float* ay,const float* az,
                        const float* bx,const float* by,const float* bz,
                        float* codex,float* cy,float* cz,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  codex[i]=ax[i]+bx[i]; cy[i]=ay[i]+by[i]; cz[i]=az[i]+bz[i];
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int n=1<<20; size_t bytes=n*sizeof(Vec3);
  Vec3 *h_a=(Vec3*)malloc(bytes), *h_b=(Vec3*)malloc(bytes), *h_c1=(Vec3*)malloc(bytes), *h_c2=(Vec3*)malloc(bytes);
  for(int i=0;i<n;i++){ h_a[i].x=i; h_a[i].y=i+1; h_a[i].z=i+2; h_b[i].x=1; h_b[i].y=2; h_b[i].z=3; }

  Vec3 *d_a,*d_b,*d_c1; CHECK(cudaMalloc(&d_a,bytes)); CHECK(cudaMalloc(&d_b,bytes)); CHECK(cudaMalloc(&d_c1,bytes));
  CHECK(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));

  float *d_ax,*d_ay,*d_az,*d_bx,*d_by,*d_bz,*d_codex,*d_cy,*d_cz;
  size_t fbytes=n*sizeof(float);
  CHECK(cudaMalloc(&d_ax,fbytes)); CHECK(cudaMalloc(&d_ay,fbytes)); CHECK(cudaMalloc(&d_az,fbytes));
  CHECK(cudaMalloc(&d_bx,fbytes)); CHECK(cudaMalloc(&d_by,fbytes)); CHECK(cudaMalloc(&d_bz,fbytes));
  CHECK(cudaMalloc(&d_codex,fbytes)); CHECK(cudaMalloc(&d_cy,fbytes)); CHECK(cudaMalloc(&d_cz,fbytes));
  float *h_ax=(float*)malloc(fbytes), *h_ay=(float*)malloc(fbytes), *h_az=(float*)malloc(fbytes);
  float *h_bx=(float*)malloc(fbytes), *h_by=(float*)malloc(fbytes), *h_bz=(float*)malloc(fbytes);
  for(int i=0;i<n;i++){ h_ax[i]=h_a[i].x; h_ay[i]=h_a[i].y; h_az[i]=h_a[i].z; h_bx[i]=h_b[i].x; h_by[i]=h_b[i].y; h_bz[i]=h_b[i].z; }
  CHECK(cudaMemcpy(d_ax,h_ax,fbytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_ay,h_ay,fbytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_az,h_az,fbytes,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_bx,h_bx,fbytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_by,h_by,fbytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_bz,h_bz,fbytes,cudaMemcpyHostToDevice));

  dim3 block(256), grid((n+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); add_aos<<<grid,block>>>(d_a,d_b,d_c1,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); add_soa<<<grid,block>>>(d_ax,d_ay,d_az,d_bx,d_by,d_bz,d_codex,d_cy,d_cz,n); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));

  CHECK(cudaMemcpy(h_c1,d_c1,bytes,cudaMemcpyDeviceToHost));
  float *h_codex=(float*)malloc(fbytes), *h_cy=(float*)malloc(fbytes), *h_cz=(float*)malloc(fbytes);
  CHECK(cudaMemcpy(h_codex,d_codex,fbytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h_cy,d_cy,fbytes,cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h_cz,d_cz,fbytes,cudaMemcpyDeviceToHost));
  for(int i=0;i<n;i++){ h_c2[i].x=h_codex[i]; h_c2[i].y=h_cy[i]; h_c2[i].z=h_cz[i]; }
  float d1=max_diff((float*)h_c1,(float*)h_c2,n*3);
  printf("max diff: %g\n", d1);
  printf("AoS: %.3f ms, SoA: %.3f ms\n", t1,t2);

  free(h_a); free(h_b); free(h_c1); free(h_c2); free(h_ax); free(h_ay); free(h_az); free(h_bx); free(h_by); free(h_bz); free(h_codex); free(h_cy); free(h_cz);
  CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c1));
  CHECK(cudaFree(d_ax)); CHECK(cudaFree(d_ay)); CHECK(cudaFree(d_az)); CHECK(cudaFree(d_bx)); CHECK(cudaFree(d_by)); CHECK(cudaFree(d_bz)); CHECK(cudaFree(d_codex)); CHECK(cudaFree(d_cy)); CHECK(cudaFree(d_cz));
  return 0;
}
