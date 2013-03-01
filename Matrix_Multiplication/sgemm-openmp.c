#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h> 
#include <omp.h>


void helper (int,  float*, float*, float*);


void helper2(int, int ,int, int, float*, float*,float*);
void sgemm_helper(int ,  int , int , int , float* , float*, float* );
void sgemm_helper2 (int,float*, float*, float*);



void sgemm_helper( int m, int a, int b, int c, float *A, float* B, float *C )
{

    for ( int k = 0; k < a; k += 8 ){
        for ( int j = 0; j < b; j += 8){
            for (int i = 0; i < c;i+=8)
            {
              
            
                if (a-k >= 8 && b-j >= 8 && c-i >= 8){
                    helper( m,  A+i+k*m, B+j+k*m, C+i+j*m );
                   
                }
                else{
                    int comp1 = (a-k < 8) ? a-k : 8;
                    int comp2 = (b-j < 8) ? b-j : 8;
                    int comp3 = (c-i < 8) ? c-i : 8;
                    helper2(m,comp1,comp2,comp3, A+i+k*m, B+j+k*m, C+i+j*m );
                
            }
        
        }
    }
}
}

void sgemm_helper2( int m, float *A, float *B, float *C )
{

    
	for(int k = 0; k < 64; k+=8)
		for(int j = 0; j < 64; j+=8)
			for(int i = 0; i < 64; i+=8)
				helper(m, A + i + k*m, B + j + k*m, C + i + j*m);
}

void sgemm( int m, int n, float *A, float *C )
{
    #pragma omp parallel
    {

#pragma omp for
        
            for ( int j = 0; j < m; j += 64){
                for ( int k = 0; k < n; k += 64 ){
                    for (int i = 0; i < m;i+=64){
                    
                    
                    if (n-k >= 64 && m-j >= 64 && m-i >= 64){
                        sgemm_helper2( m, A+i+k*m, A+j+k*m, C+i+j*m );
                    
                    }
                    else{
                        int comp1 = (n-k < 64) ? n-k : 64;
                        int comp2 = (m-j < 64) ? m-j : 64;
                        int comp3 = (m-i < 64) ? m-i : 64;
                        sgemm_helper(m,comp1,comp2,comp3, A+i+k*m, A+j+k*m, C+i+j*m );
        
                        
                    }
                    
                }
            }
            }


	}
}


    
void helper2(int m, int a, int b, int c, float *A, float *B, float *C)
{
    for(int k = 0; k<a; k++)
        for (int j = 0; j<b; j++)
        {
            for( int i = 0; i < c; i++ )
            {
                *(C + i + j*m) += A[i+k*m] * B[j+k*m];
            }
        }
}


void helper(int m, float *A, float *B, float *C)
{
    __m128 A11 = _mm_loadu_ps(A);
	A+=4;
	__m128 A12 = _mm_loadu_ps(A);
	A+=(m-4);
	__m128 A21 = _mm_loadu_ps(A);
	A+=4;
	__m128 A22 = _mm_loadu_ps(A);
	A+=(m-4);
	__m128 A31 = _mm_loadu_ps(A);
	A+=4;
	__m128 A32 = _mm_loadu_ps(A);
	A+=(m-4);
	__m128 A41 = _mm_loadu_ps(A);
	A+=4;
	__m128 A42 = _mm_loadu_ps(A);
	A+=(m-4);
	

    __m128 B1 = _mm_load1_ps(B);
	__m128 B2 = _mm_load1_ps(B+m);
	__m128 B3 = _mm_load1_ps(B+2*m);
	__m128 B4 = _mm_load1_ps(B+3*m);
	__m128 C1 = _mm_loadu_ps(C);
	

	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	

	C1 = _mm_loadu_ps(C+4);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	
	

	B++;
	B1 = _mm_load1_ps(B);
	B2 = _mm_load1_ps(B+m);
	B3 = _mm_load1_ps(B+2*m);
	B4 = _mm_load1_ps(B+3*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	
	
	B++;
	B1 = _mm_load1_ps(B);
	B2 = _mm_load1_ps(B+m);
	B3 = _mm_load1_ps(B+2*m);
	B4 = _mm_load1_ps(B+3*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	

    B++;
	B1 = _mm_load1_ps(B);
	B2 = _mm_load1_ps(B+m);
	B3 = _mm_load1_ps(B+2*m);
	B4 = _mm_load1_ps(B+3*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	

    B++;
	B1 = _mm_load1_ps(B);
	B2 = _mm_load1_ps(B+m);
	B3 = _mm_load1_ps(B+2*m);
	B4 = _mm_load1_ps(B+3*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	

    B++;
	B1 = _mm_load1_ps(B);
	B2 = _mm_load1_ps(B+m);
	B3 = _mm_load1_ps(B+2*m);
	B4 = _mm_load1_ps(B+3*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	

    B++;
	B1 = _mm_load1_ps(B);
	B2 = _mm_load1_ps(B+m);
	B3 = _mm_load1_ps(B+2*m);
	B4 = _mm_load1_ps(B+3*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	

    B++;
	B1 = _mm_load1_ps(B);
	B2 = _mm_load1_ps(B+m);
	B3 = _mm_load1_ps(B+2*m);
	B4 = _mm_load1_ps(B+3*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	
	
	A11 = _mm_loadu_ps(A);
	A+=4;
	A12 = _mm_loadu_ps(A);
	A+=(m-4);
	A21 = _mm_loadu_ps(A);
	A+=4;
	A22 = _mm_loadu_ps(A);
	A+=(m-4);
	A31 = _mm_loadu_ps(A);
	A+=4;
	A32 = _mm_loadu_ps(A);
	A+=(m-4);
	A41 = _mm_loadu_ps(A);
	A+=4;
	A42 = _mm_loadu_ps(A);
	

	C-=7*m;
	B-=7;
	C1 = _mm_loadu_ps(C);

	B1 = _mm_load1_ps(B+4*m);
	B2 = _mm_load1_ps(B+5*m);
	B3 = _mm_load1_ps(B+6*m);
	B4 = _mm_load1_ps(B+7*m);

	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	

	C1 = _mm_loadu_ps(C+4);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	
	
	B++;
	B1 = _mm_load1_ps(B+4*m);
	B2 = _mm_load1_ps(B+5*m);
	B3 = _mm_load1_ps(B+6*m);
	B4 = _mm_load1_ps(B+7*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
    C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	
	
	B++;
	B1 = _mm_load1_ps(B+4*m);
	B2 = _mm_load1_ps(B+5*m);
	B3 = _mm_load1_ps(B+6*m);
	B4 = _mm_load1_ps(B+7*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
    C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	
	
	B++;
	B1 = _mm_load1_ps(B+4*m);
	B2 = _mm_load1_ps(B+5*m);
	B3 = _mm_load1_ps(B+6*m);
	B4 = _mm_load1_ps(B+7*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
    C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);

	B++;
	B1 = _mm_load1_ps(B+4*m);
	B2 = _mm_load1_ps(B+5*m);
	B3 = _mm_load1_ps(B+6*m);
	B4 = _mm_load1_ps(B+7*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
    C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	

	B++;
	B1 = _mm_load1_ps(B+4*m);
	B2 = _mm_load1_ps(B+5*m);
	B3 = _mm_load1_ps(B+6*m);
	B4 = _mm_load1_ps(B+7*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
    C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	

	B++;
	B1 = _mm_load1_ps(B+4*m);
	B2 = _mm_load1_ps(B+5*m);
	B3 = _mm_load1_ps(B+6*m);
	B4 = _mm_load1_ps(B+7*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
    C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
	
	
	
	B++;
	B1 = _mm_load1_ps(B+4*m);
	B2 = _mm_load1_ps(B+5*m);
	B3 = _mm_load1_ps(B+6*m);
	B4 = _mm_load1_ps(B+7*m);
	C+=m;
	C1 = _mm_loadu_ps(C);
	C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A11));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A21));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A31));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A41));
	_mm_storeu_ps(C, C1);
	C1 = _mm_loadu_ps(C+4);
    C1 = _mm_add_ps(C1, _mm_mul_ps(B1, A12));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B2, A22));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B3, A32));
	C1 = _mm_add_ps(C1, _mm_mul_ps(B4, A42));
	_mm_storeu_ps((C+4), C1);
}



