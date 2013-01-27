#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h> 




void sgemm( int m, int n, float *A, float *C )
{
    __m128 A1;
    __m128 A2;
    __m128 A3;
    __m128 A4;
    
    __m128 B1;
    __m128 B2;
    __m128 B3;
    __m128 B4;
    
    __m128 C_1;
    __m128 C_2;
    __m128 C_3;
    __m128 C_4;
    
    int offset1, offset2;
    
    // preparations
    int mul = 4;                    // block_size;
    int padding_flag = 0;
    float* pt = A;
    float* largeC;

    
    // padding
    int compm = (m % mul == 0) ? m : (m/mul+1)*mul;
    int compn = (n % mul == 0) ? n : (n/mul+1)*mul;

    if (compm > m || compn > n) {
        padding_flag = 1;
        largeC = (float*) calloc (compm * compm, sizeof(float));
    } else {
        largeC = C;
        pt = A;
    }
    
    if (padding_flag){
        pt = (float*) calloc(compm * compn, sizeof(float));         //
        for (int p = 0; p < n; p++){                                //  can be unrolled.
            memcpy(pt+p*compm, A+p*m, sizeof(float) * m);           //
        }
        A = pt;
    }

    

    // Multiplying matrices
    for ( int j = 0; j < compm; j += mul ){
        
        for (int i = 0; i < compm; i += mul ) {
            offset1 = i+j*compm;
            C_1 = _mm_loadu_ps( largeC + offset1 );
            C_2 = _mm_loadu_ps( largeC + offset1 + compm );
            C_3 = _mm_loadu_ps( largeC + offset1 + 2 * compm );
            C_4 = _mm_loadu_ps( largeC + offset1 + 3 * compm );
        
            for ( int k = 0; k < compn; k += mul ){
                offset2 = j+k*compm;
                A1 = _mm_loadu_ps( pt+ i + k * compm );
                A2 = _mm_loadu_ps( pt+ i + ( k + 1 ) * compm );
                A3 = _mm_loadu_ps( pt+ i + ( k + 2 ) * compm );
                A4 = _mm_loadu_ps( pt+ i + ( k + 3 ) * compm );
                
                B1 = _mm_load1_ps( pt+ offset2 );
				B2 = _mm_load1_ps( pt+ offset2+compm );
				B3 = _mm_load1_ps( pt+ offset2+2*compm );
				B4 = _mm_load1_ps( pt+ offset2+3*compm );
                
                C_1 = _mm_add_ps( C_1, _mm_mul_ps( B1, A1 ) );
                C_1 = _mm_add_ps( C_1, _mm_mul_ps( B2, A2 ) );
                C_1 = _mm_add_ps( C_1, _mm_mul_ps( B3, A3 ) );
                C_1 = _mm_add_ps( C_1, _mm_mul_ps( B4, A4 ) );
                
                B1 = _mm_load1_ps( pt+ offset2 + 1 );
				B2 = _mm_load1_ps( pt+ offset2+compm + 1 );
				B3 = _mm_load1_ps( pt+ offset2+2*compm + 1 );
				B4 = _mm_load1_ps( pt+ offset2+3*compm + 1 );
                
                C_2 = _mm_add_ps( C_2, _mm_mul_ps( B1, A1 ) );
                C_2 = _mm_add_ps( C_2, _mm_mul_ps( B2, A2 ) );
                C_2 = _mm_add_ps( C_2, _mm_mul_ps( B3, A3 ) );
                C_2 = _mm_add_ps( C_2, _mm_mul_ps( B4, A4 ) );
                
                B1 = _mm_load1_ps( pt+ offset2 + 2);
				B2 = _mm_load1_ps( pt+ offset2+compm + 2);
				B3 = _mm_load1_ps( pt+ offset2+2*compm + 2);
				B4 = _mm_load1_ps( pt+ offset2+3*compm + 2);
                
                C_3 = _mm_add_ps( C_3, _mm_mul_ps( B1, A1 ) );
                C_3 = _mm_add_ps( C_3, _mm_mul_ps( B2, A2 ) );
                C_3 = _mm_add_ps( C_3, _mm_mul_ps( B3, A3 ) );
                C_3 = _mm_add_ps( C_3, _mm_mul_ps( B4, A4 ) );
                
                B1 = _mm_load1_ps( pt+ offset2 + 3);
				B2 = _mm_load1_ps( pt+ offset2+compm + 3);
				B3 = _mm_load1_ps( pt+ offset2+2*compm + 3);
				B4 = _mm_load1_ps( pt+ offset2+3*compm + 3);
                
                C_4 = _mm_add_ps( C_4, _mm_mul_ps( B1, A1 ) );
                C_4 = _mm_add_ps( C_4, _mm_mul_ps( B2, A2 ) );
                C_4 = _mm_add_ps( C_4, _mm_mul_ps( B3, A3 ) );
                C_4 = _mm_add_ps( C_4, _mm_mul_ps( B4, A4 ) );
            }
            
            _mm_storeu_ps( largeC + offset1, C_1 );
            _mm_storeu_ps( largeC + offset1 + compm, C_2 );
            _mm_storeu_ps( largeC + offset1 + 2 * compm, C_3 );
            _mm_storeu_ps( largeC + offset1 + 3 * compm, C_4 );
        }
    }
                

    
    
    // truncating
    if (padding_flag == 1){
        //        printf("\nstart truncating. . . ");
        for ( int j = 0; j < m; j++ )
        {
            // this 10 is non-deterministic
            int bound = m/10*10;
            for ( int i = 0; i < bound; i+=10 )
            {
                int offset1 = i+j*m;
                int offset2 = i+j*compm;
                C[offset1] = largeC[offset2];
                C[offset1+1] = largeC[offset2+1];
                C[offset1+2] = largeC[offset2+2];
                C[offset1+3] = largeC[offset2+3];
                C[offset1+4] = largeC[offset2+4];
                C[offset1+5] = largeC[offset2+5];
                C[offset1+6] = largeC[offset2+6];
                C[offset1+7] = largeC[offset2+7];
                C[offset1+8] = largeC[offset2+8];
                C[offset1+9] = largeC[offset2+9];
            }
            
            for ( int i = bound; i < m; i++ )
            {
                C[i+j*m] = largeC[i+j*compm];
            }
        }
        free(largeC);
    }
}


