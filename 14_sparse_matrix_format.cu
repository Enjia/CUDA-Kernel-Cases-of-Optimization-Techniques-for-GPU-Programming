/*
注意事项：
- 格式转换成本需评估
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CHECK(x) do{cudaError_t _err=(x); if(_err!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} }while(0)

__global__ void spmv_coo(const int* row, const int* col, const float* val, const float* x, float* y, int nnz){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nnz) return;
  atomicAdd(&y[row[i]], val[i]*x[col[i]]);
}


// 优化点：选择合适稀疏格式以匹配访问模式与并行性
__global__ void spmv_csr(const int* rowPtr, const int* col, const float* val, const float* x, float* y, int rows){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=rows) return;
  float sum=0.f; for(int i=rowPtr[r]; i<rowPtr[r+1]; i++) sum+=val[i]*x[col[i]]; y[r]=sum;
}

float max_diff(const float* a, const float* b, int n){
  float m=0.f; for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m) m=d;} return m; }

int main(){
  int rows=1024, cols=1024, nnz=rows*4;
  int *h_row=(int*)malloc(nnz*sizeof(int)); int *h_col=(int*)malloc(nnz*sizeof(int)); float *h_val=(float*)malloc(nnz*sizeof(float));
  for(int r=0;r<rows;r++) for(int k=0;k<4;k++){ int idx=r*4+k; h_row[idx]=r; h_col[idx]=(r*7+k*13)%cols; h_val[idx]=0.1f*(k+1); }
  float *h_x=(float*)malloc(cols*sizeof(float)), *h_y1=(float*)malloc(rows*sizeof(float)), *h_y2=(float*)malloc(rows*sizeof(float));
  for(int i=0;i<cols;i++) h_x[i]=sinf(i*0.01f);

  int *h_rowPtr=(int*)malloc((rows+1)*sizeof(int)); h_rowPtr[0]=0; for(int r=0;r<rows;r++) h_rowPtr[r+1]=h_rowPtr[r]+4;

  int *d_row,*d_col,*d_rowPtr; float *d_val,*d_x,*d_y1,*d_y2;
  CHECK(cudaMalloc(&d_row,nnz*sizeof(int))); CHECK(cudaMalloc(&d_col,nnz*sizeof(int))); CHECK(cudaMalloc(&d_val,nnz*sizeof(float)));
  CHECK(cudaMalloc(&d_rowPtr,(rows+1)*sizeof(int))); CHECK(cudaMalloc(&d_x,cols*sizeof(float))); CHECK(cudaMalloc(&d_y1,rows*sizeof(float))); CHECK(cudaMalloc(&d_y2,rows*sizeof(float)));
  CHECK(cudaMemcpy(d_row,h_row,nnz*sizeof(int),cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_col,h_col,nnz*sizeof(int),cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_val,h_val,nnz*sizeof(float),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_rowPtr,h_rowPtr,(rows+1)*sizeof(int),cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_x,h_x,cols*sizeof(float),cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_y1,0,rows*sizeof(float)));

  dim3 block(256), grid1((nnz+block.x-1)/block.x), grid2((rows+block.x-1)/block.x);
  cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  CHECK(cudaEventRecord(s)); spmv_coo<<<grid1,block>>>(d_row,d_col,d_val,d_x,d_y1,nnz); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t1; CHECK(cudaEventElapsedTime(&t1,s,e));
  CHECK(cudaEventRecord(s)); spmv_csr<<<grid2,block>>>(d_rowPtr,d_col,d_val,d_x,d_y2,rows); CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
  float t2; CHECK(cudaEventElapsedTime(&t2,s,e));

  CHECK(cudaMemcpy(h_y1,d_y1,rows*sizeof(float),cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h_y2,d_y2,rows*sizeof(float),cudaMemcpyDeviceToHost));
  printf("max diff: %g\n", max_diff(h_y1,h_y2,rows));
  printf("COO: %.3f ms, CSR: %.3f ms\n", t1,t2);

  free(h_row); free(h_col); free(h_val); free(h_x); free(h_y1); free(h_y2); free(h_rowPtr);
  CHECK(cudaFree(d_row)); CHECK(cudaFree(d_col)); CHECK(cudaFree(d_val)); CHECK(cudaFree(d_rowPtr)); CHECK(cudaFree(d_x)); CHECK(cudaFree(d_y1)); CHECK(cudaFree(d_y2));
  return 0;
}
