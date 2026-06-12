#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"
#include "kernel.h"
#include <omp.h>
#include <stdbool.h>
#include "dnnl.h"
#include <string.h>
#include <omp.h>


void create_matrix_A_31(int8_t* A, int m, int n) {
    srand(time(NULL));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            A[j * m + i] = (int8_t)((rand() % 32) - 16);  
            
}

void create_matrix_B_31(int8_t* A, int m, int n) {
    srand(time(NULL)*2);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            A[j * m + i] = (int8_t)((rand() % 32) - 16);  
            
}

// when using this onednn gets wrong answers
void create_matrix_A_127(int8_t* A, int m, int n) {
    srand(time(NULL));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            A[j * m + i] = (int8_t)((rand() % 256) - 128); 
}
void create_matrix_B_127(int8_t* A, int m, int n) {
    srand(time(NULL));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            A[j * m + i] = (int8_t)((rand() % 256) - 128); 
}

void naive_matmul(int8_t* A, int8_t* B, int32_t* C, int m, int n, int k) {
    int overflow = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int32_t acc = 0;
            for (int v = 0; v < k; v++) {
                acc += B[v * m + i] * A[j * k + v];  
            }
            C[j * m + i] = acc; 
            }                      
        }
    }

bool Check_matrix(int32_t *A, int32_t *B, int n){

    int32_t diff = 0;
    int i;

    for (i = 0; B + i && A + i && i < n; i++){
        diff = abs(A[i] - B[i]);
        if (diff > 0) {
            printf("error. %5.2d,%5.2d,%d\n", A[i], B[i], i);
            return false;
        }
    }
    return true;
}

void main(int LDB,int LDC,int LDA) {
    
    for (int i = 1000; i <= 5000; i += 1000) {
        
        int m = i, n = i, k = i;
        int8_t ao = 0, bo = 0;
        int32_t oc = 0;
        int8_t* A = (int8_t*)malloc(m * k * sizeof(int8_t));
        int8_t* B = (int8_t*)malloc(k * n * sizeof(int8_t));
        int32_t* C_dnnl = (int32_t*)malloc(m * n * sizeof(int32_t));
        int32_t* C_naive = (int32_t*)malloc(m * n * sizeof(int32_t));
        int32_t* C_kernel = (int32_t*)malloc(m * n * sizeof(int32_t));

    create_matrix_A_31(A, m, k);
    create_matrix_B_31(B, k, n);
    
    memset(C_dnnl, 0, m * n * sizeof(int32_t));
    memset(C_naive, 0, m * n * sizeof(int32_t));
    memset(C_kernel, 0, m * n * sizeof(int32_t));

    dnnl_gemm_s8s8s32('N','N','F', m,n,k, 1.0f, A,m,ao, B,k,bo, 0.0f, C_dnnl,m,&oc);
 
    kernel(m, n, k, B, m, A, k, C_kernel, m);

    naive_matmul(A, B, C_naive, m, n, k);

    printf("Correctence run aginst Naive trusted kernel:\n");

    printf("[[[[[ %d x %d x %d ]]]]]\n", m, n, k);
    if (!Check_matrix((int32_t*)C_dnnl, (int32_t*)C_naive, m * n))
        printf("====onednn-Failed-aginst-naive====\n");
    else
        printf("====onednn-Passed-aginst-naive====\n");

    if (!Check_matrix((int32_t*)C_kernel, (int32_t*)C_naive, m * n))
        printf("====Kernel-Failed-aginst-naive====\n\n");
    else
        printf("====Kernel-Passed-aginst-naive====\n\n");


    free(A);
    free(B);
    free(C_dnnl);
    free(C_naive);
    free(C_kernel);  
}
}
