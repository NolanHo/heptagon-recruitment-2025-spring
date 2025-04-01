#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include "utils.h"
#include <time.h>
#include <omp.h>
#include <algorithm>
// #include <jemalloc/jemalloc.h>

#include "flag.h"


inline int64_t current_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)(ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL);
}

inline int64_t current_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)(ts.tv_sec * 1000000000LL + ts.tv_nsec);
}



void filter_transform(float *__restrict__ filter,
                      float *__restrict__ V,
                      const filter_shape_t fs,
                      const U_shape_t us,
                      const int64_t collapsed_dim_size) {
  typedef float (*filter_tensor_t)[fs.ic][fs.h][fs.w];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;

  typedef float (*V_tensor_t)[us.w][collapsed_dim_size];
  V_tensor_t V_tensor = (V_tensor_t)V;

  static const __m128 v1_4      = _mm_set1_ps(1.0f / 4.0f);
  static const __m128 v_neg1_6  = _mm_set1_ps(-1.0f / 6.0f);
  static const __m128 v1_6      = _mm_set1_ps(1.0f / 6.0f);
  static const __m128 v1_24     = _mm_set1_ps(1.0f / 24.0f);
  static const __m128 v1_12     = _mm_set1_ps(1.0f / 12.0f);
  static const __m128 v_neg1_12 = _mm_set1_ps(-1.0f / 12.0f);

  #pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    int oc     = (int)(idx / fs.ic);
    int ic_idx = (int)(idx % fs.ic);

    __m128 g0 = _mm_setr_ps(filter_tensor[oc][ic_idx][0][0],
                            filter_tensor[oc][ic_idx][0][1],
                            filter_tensor[oc][ic_idx][0][2],
                            0.0f);
    __m128 g1 = _mm_setr_ps(filter_tensor[oc][ic_idx][1][0],
                            filter_tensor[oc][ic_idx][1][1],
                            filter_tensor[oc][ic_idx][1][2],
                            0.0f);
    __m128 g2 = _mm_setr_ps(filter_tensor[oc][ic_idx][2][0],
                            filter_tensor[oc][ic_idx][2][1],
                            filter_tensor[oc][ic_idx][2][2],
                            0.0f);

    __m128 U0 = _mm_mul_ps(g0, v1_4);

    __m128 sum = _mm_add_ps(_mm_add_ps(g0, g1), g2);
    __m128 U1 = _mm_mul_ps(sum, v_neg1_6);

    __m128 U2 = _mm_add_ps(
      _mm_mul_ps(g0, v_neg1_6),
      _mm_add_ps(
        _mm_mul_ps(g1, v1_6),
        _mm_mul_ps(g2, v_neg1_6)
      )
    );

    __m128 U3 = _mm_add_ps(
      _mm_mul_ps(g0, v1_24),
      _mm_add_ps(
        _mm_mul_ps(g1, v1_12),
        _mm_mul_ps(g2, v1_6)
      )
    );

    __m128 U4 = _mm_add_ps(
      _mm_mul_ps(g0, v1_24),
      _mm_add_ps(
        _mm_mul_ps(g1, v_neg1_12),
        _mm_mul_ps(g2, v1_6)
      )
    );

    __m128 U5 = g2;
    float V_local[6][6];

    auto transform_row = [&](const __m128 &u, float v[6]) {
      __m128 u0 = _mm_shuffle_ps(u, u, _MM_SHUFFLE(0,0,0,0));
      __m128 u1 = _mm_shuffle_ps(u, u, _MM_SHUFFLE(1,1,1,1));
      __m128 u2 = _mm_shuffle_ps(u, u, _MM_SHUFFLE(2,2,2,2));

      __m128 r0 = _mm_mul_ps(u0, v1_4);
      __m128 sum_u = _mm_add_ps(_mm_add_ps(u0, u1), u2);
      __m128 r1 = _mm_mul_ps(sum_u, v_neg1_6);
      __m128 r2 = _mm_add_ps(
                   _mm_mul_ps(u0, v_neg1_6),
                   _mm_add_ps(
                     _mm_mul_ps(u1, v1_6),
                     _mm_mul_ps(u2, v_neg1_6)
                   )
                 );
      __m128 r3 = _mm_add_ps(
                   _mm_mul_ps(u0, v1_24),
                   _mm_add_ps(
                     _mm_mul_ps(u1, v1_12),
                     _mm_mul_ps(u2, v1_6)
                   )
                 );
      __m128 r4 = _mm_add_ps(
                   _mm_mul_ps(u0, v1_24),
                   _mm_add_ps(
                     _mm_mul_ps(u1, v_neg1_12),
                     _mm_mul_ps(u2, v1_6)
                   )
                 );
      __m128 r5 = u2;

      v[0] = _mm_cvtss_f32(r0);
      v[1] = _mm_cvtss_f32(r1);
      v[2] = _mm_cvtss_f32(r2);
      v[3] = _mm_cvtss_f32(r3);
      v[4] = _mm_cvtss_f32(r4);
      v[5] = _mm_cvtss_f32(r5);
    };

    transform_row(U0, V_local[0]);
    transform_row(U1, V_local[1]);
    transform_row(U2, V_local[2]);
    transform_row(U3, V_local[3]);
    transform_row(U4, V_local[4]);
    transform_row(U5, V_local[5]);

    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        V_tensor[i][j][idx] = V_local[i][j];
      }
    }
  }
}

inline void image_transform_simd(float local_tile[6][6],
                                 float final_tile[6][6]) {
    float V_local[6][6];
    for (int w = 0; w < 4; w += 4) {
        __m128 r0 = _mm_loadu_ps(&local_tile[0][w]);
        __m128 r1 = _mm_loadu_ps(&local_tile[1][w]);
        __m128 r2 = _mm_loadu_ps(&local_tile[2][w]);
        __m128 r3 = _mm_loadu_ps(&local_tile[3][w]);
        __m128 r4 = _mm_loadu_ps(&local_tile[4][w]);
        __m128 r5 = _mm_loadu_ps(&local_tile[5][w]);

        __m128 v0 = _mm_add_ps(
                        _mm_mul_ps(_mm_set1_ps(4.f), r0),
                        _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-5.f), r2), r4)
                      );
        _mm_storeu_ps(&V_local[0][w], v0);

        __m128 v1 = _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-4.f), r1),
                                   _mm_mul_ps(_mm_set1_ps(-4.f), r2)),
                        _mm_add_ps(r3, r4)
                      );
        _mm_storeu_ps(&V_local[1][w], v1);

        __m128 v2 = _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(_mm_set1_ps(4.f), r1),
                                   _mm_mul_ps(_mm_set1_ps(-4.f), r2)),
                        _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-1.f), r3), r4)
                      );
        _mm_storeu_ps(&V_local[2][w], v2);

        __m128 v3 = _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-2.f), r1),
                                   _mm_mul_ps(_mm_set1_ps(-1.f), r2)),
                        _mm_add_ps(_mm_mul_ps(_mm_set1_ps(2.f), r3), r4)
                      );
        _mm_storeu_ps(&V_local[3][w], v3);

        __m128 v4 = _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(_mm_set1_ps(2.f), r1),
                                   _mm_mul_ps(_mm_set1_ps(-1.f), r2)),
                        _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-2.f), r3), r4)
                      );
        _mm_storeu_ps(&V_local[4][w], v4);

        __m128 v5 = _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(_mm_set1_ps(4.f), r1),
                                   _mm_mul_ps(_mm_set1_ps(-5.f), r3)),
                        r5
                      );
        _mm_storeu_ps(&V_local[5][w], v5);
    }
    for (int w = 4; w < 6; w++) {
        float a0 = local_tile[0][w];
        float a1 = local_tile[1][w];
        float a2 = local_tile[2][w];
        float a3 = local_tile[3][w];
        float a4 = local_tile[4][w];
        float a5 = local_tile[5][w];
        V_local[0][w] = 4.f * a0 + (-5.f) * a2 + a4;
        V_local[1][w] = -4.f * a1 + (-4.f) * a2 + a3 + a4;
        V_local[2][w] = 4.f * a1 + (-4.f) * a2 + (-1.f)*a3 + a4;
        V_local[3][w] = -2.f * a1 + (-1.f)*a2 + 2.f * a3 + a4;
        V_local[4][w] = 2.f * a1 + (-1.f)*a2 + (-2.f)*a3 + a4;
        V_local[5][w] = 4.f * a1 + (-5.f)*a3 + a5;
    }

    for (int h = 0; h < 6; h++) {
        __m128 vec = _mm_loadu_ps(&V_local[h][0]);
        float v4 = V_local[h][4];
        float v5 = V_local[h][5];

        __m128 coeff0 = _mm_set_ps(0.f, -5.f, 0.f, 4.f);
        float t0 = _mm_cvtss_f32(_mm_dp_ps(vec, coeff0, 0xF1)) + v4;

        __m128 coeff1 = _mm_set_ps(1.f, -4.f, -4.f, 0.f);
        float t1 = _mm_cvtss_f32(_mm_dp_ps(vec, coeff1, 0xF1)) + v4;

        __m128 coeff2 = _mm_set_ps(-1.f, -4.f, 4.f, 0.f);
        float t2 = _mm_cvtss_f32(_mm_dp_ps(vec, coeff2, 0xF1)) + v4;

        __m128 coeff3 = _mm_set_ps(2.f, -1.f, -2.f, 0.f);
        float t3 = _mm_cvtss_f32(_mm_dp_ps(vec, coeff3, 0xF1)) + v4;

        __m128 coeff4 = _mm_set_ps(-2.f, -1.f, 2.f, 0.f);
        float t4 = _mm_cvtss_f32(_mm_dp_ps(vec, coeff4, 0xF1)) + v4;

        __m128 coeff5 = _mm_set_ps(-5.f, 0.f, 4.f, 0.f);
        float t5 = _mm_cvtss_f32(_mm_dp_ps(vec, coeff5, 0xF1)) + v5;

        final_tile[h][0] = t0;
        final_tile[h][1] = t1;
        final_tile[h][2] = t2;
        final_tile[h][3] = t3;
        final_tile[h][4] = t4;
        final_tile[h][5] = t5;
    }
}

void image_transform_reverse(float *__restrict__ image, // [batch][input_channel][input_height][input_width]
                      float *__restrict__ V,
                      const image_shape_t is,
                      const V_shape_t vs,
                      const tiling_info_t ti) {
  int64_t collapsed_dim_size = vs.ic * ti.num_tiles;
  typedef float (*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float (*image_tensor_t)[is.ic][is.h][is.w];
  image_tensor_t image_tensor = (image_tensor_t)image;
  V_tensor_t V_tensor = (V_tensor_t)V;

  // V[tile_in_h][tile_in_w][ic][num_tiles]

  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t tile_idx = 0; tile_idx < ti.num_tiles; tile_idx++) {
    for (int64_t ic_idx = 0; ic_idx < is.ic; ic_idx++) {
      tile_index_t tidx = get_tile_index(tile_idx, ti);
      int batch = tidx.b;
      int base_h = tidx.th * 4;
      int base_w = tidx.tw * 4;

      // tile_in_h, ti.tile_in_w
      float local_tile[6][6];
      for (int h = 0; h < 6; h++) {
        for (int w = 0; w < 6; w++) {
          int img_h = base_h + h;
          int img_w = base_w + w;
          if (img_h < is.h && img_w < is.w) {
            local_tile[h][w] = image_tensor[batch][ic_idx][img_h][img_w];
          } else {
            local_tile[h][w] = 0.0f;
          }
        }
      }

      float final_tile[6][6];
      image_transform_simd(local_tile, final_tile);

      int64_t idx = ic_idx * ti.num_tiles + tile_idx;
      // int64_t idx = tile_idx * is.ic + ic_idx;
      for (int h = 0; h < ti.tile_in_h; h++) {
        for (int w = 0; w < ti.tile_in_w; w++) {
          V_tensor[h][w][idx] = final_tile[h][w];
        }
      }
    }
  }
  // #pragma omp parallel for schedule(static)
  // for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
  //   // V[tile_in_h][tile_in_w][num_tiles][ic]
  //   // int tile_idx = idx / is.ic;
  //   // int ic   = idx % is.ic;

  //   // V[tile_in_h][tile_in_w][ic][num_tiles]
  //   int ic = idx / ti.num_tiles;
  //   int tile_idx = idx % ti.num_tiles;

  //   tile_index_t tidx = get_tile_index(tile_idx, ti);
  //   int batch = tidx.b;
  //   int base_h = tidx.th * 4;
  //   int base_w = tidx.tw * 4;

  //   // tile_in_h, ti.tile_in_w
  //   float local_tile[6][6];
  //   for (int h = 0; h < 6; h++) {
  //     for (int w = 0; w < 6; w++) {
  //       int img_h = base_h + h;
  //       int img_w = base_w + w;
  //       if (img_h < is.h && img_w < is.w) {
  //         local_tile[h][w] = image_tensor[batch][ic][img_h][img_w];
  //       } else {
  //         local_tile[h][w] = 0.0f;
  //       }
  //     }
  //   }

  //   float final_tile[6][6];
  //   image_transform_simd(local_tile, final_tile);

  //   for (int h = 0; h < ti.tile_in_h; h++) {
  //     for (int w = 0; w < 6; w++) {
  //       V_tensor[h][w][idx] = final_tile[h][w];
  //     }
  //   }
  // }
}

void image_transform(float *__restrict__ image, // [batch][input_channel][input_height][input_width]
                      float *__restrict__ V,
                      const image_shape_t is,
                      const V_shape_t vs,
                      const tiling_info_t ti) {
  int64_t collapsed_dim_size = vs.ic * ti.num_tiles;
  typedef float (*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float (*image_tensor_t)[is.ic][is.h][is.w];
  image_tensor_t image_tensor = (image_tensor_t)image;
  V_tensor_t V_tensor = (V_tensor_t)V;

  // V[tile_in_h][tile_in_w][ic][num_tiles]

  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t tile_idx = 0; tile_idx < ti.num_tiles; tile_idx++) {
    for (int64_t ic_idx = 0; ic_idx < is.ic; ic_idx++) {
      tile_index_t tidx = get_tile_index(tile_idx, ti);
      int batch = tidx.b;
      int base_h = tidx.th * 4;
      int base_w = tidx.tw * 4;

      // tile_in_h, ti.tile_in_w
      float local_tile[6][6];
      for (int h = 0; h < 6; h++) {
        for (int w = 0; w < 6; w++) {
          int img_h = base_h + h;
          int img_w = base_w + w;
          if (img_h < is.h && img_w < is.w) {
            local_tile[h][w] = image_tensor[batch][ic_idx][img_h][img_w];
          } else {
            local_tile[h][w] = 0.0f;
          }
        }
      }

      float final_tile[6][6];
      image_transform_simd(local_tile, final_tile);

      // int64_t idx = ic_idx * ti.num_tiles + tile_idx;
      int64_t idx = tile_idx * is.ic + ic_idx;
      for (int h = 0; h < ti.tile_in_h; h++) {
        for (int w = 0; w < ti.tile_in_w; w++) {
          V_tensor[h][w][idx] = final_tile[h][w];
        }
      }
    }
  }
  // #pragma omp parallel for schedule(static)
  // for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
  //   // V[tile_in_h][tile_in_w][num_tiles][ic]
  //   // int tile_idx = idx / is.ic;
  //   // int ic   = idx % is.ic;

  //   // V[tile_in_h][tile_in_w][ic][num_tiles]
  //   int ic = idx / ti.num_tiles;
  //   int tile_idx = idx % ti.num_tiles;

  //   tile_index_t tidx = get_tile_index(tile_idx, ti);
  //   int batch = tidx.b;
  //   int base_h = tidx.th * 4;
  //   int base_w = tidx.tw * 4;

  //   // tile_in_h, ti.tile_in_w
  //   float local_tile[6][6];
  //   for (int h = 0; h < 6; h++) {
  //     for (int w = 0; w < 6; w++) {
  //       int img_h = base_h + h;
  //       int img_w = base_w + w;
  //       if (img_h < is.h && img_w < is.w) {
  //         local_tile[h][w] = image_tensor[batch][ic][img_h][img_w];
  //       } else {
  //         local_tile[h][w] = 0.0f;
  //       }
  //     }
  //   }

  //   float final_tile[6][6];
  //   image_transform_simd(local_tile, final_tile);

  //   for (int h = 0; h < ti.tile_in_h; h++) {
  //     for (int w = 0; w < 6; w++) {
  //       V_tensor[h][w][idx] = final_tile[h][w];
  //     }
  //   }
  // }
}

// 固定tile：
// ts.tile_in_h = TILE_IN_H; = 6
// ts.tile_in_w = TILE_IN_W; = 6
// ts.tile_out_h = TILE_OUT_H; = 4
// ts.tile_out_w = TILE_OUT_W; = 4
#define OUTPUT_PIPELINED_FIXED_TILE

static const float output_transform_martrix[4][6] = {
  {1.f,  1.f,  1.f,  1.f,  1.f,  0.f},
  {0.f,  1.f, -1.f,  2.f, -2.f,  0.f},
  {0.f,  1.f,  1.f,  4.f,  4.f,  0.f},
  {0.f,  1.f, -1.f,  8.f, -8.f,  1.f}
};
inline void output_transform_simd(const float *local_M, // [tile_in_h][tile_in_w][output_channel][num_tiles]
                                  float *local_out) { // [batch][output_channel][output_height][output_width]
    float tempH[4][6];
    int j, k, i;

    for (j = 0; j < 6; j++) {
        __m128 acc = _mm_setzero_ps();
        for (k = 0; k < 6; k++) {
            float m_val = local_M[k * 6 + j];  
            __m128 m_val_vec = _mm_set1_ps(m_val);
            __m128 hvec = _mm_setr_ps(
                output_transform_martrix[0][k],
                output_transform_martrix[1][k],
                output_transform_martrix[2][k],
                output_transform_martrix[3][k]
            );
            acc = _mm_fmadd_ps(m_val_vec, hvec, acc);
        }
        float acc_arr[4];
        _mm_storeu_ps(acc_arr, acc);
        tempH[0][j] = acc_arr[0];
        tempH[1][j] = acc_arr[1];
        tempH[2][j] = acc_arr[2];
        tempH[3][j] = acc_arr[3];
    }
    for (i = 0; i < 4; i++) {
        __m128 acc = _mm_setzero_ps();
        for (k = 0; k < 6; k++) {
            float t_val = tempH[i][k];  // tempH[i][k]
            __m128 t_vec = _mm_set1_ps(t_val);
            __m128 hvec = _mm_setr_ps(
                output_transform_martrix[0][k],
                output_transform_martrix[1][k],
                output_transform_martrix[2][k],
                output_transform_martrix[3][k]
            );
            acc = _mm_fmadd_ps(t_vec, hvec, acc);
        }
        _mm_storeu_ps(&local_out[i * 4], acc);
    }
}


void output_transform(float *__restrict__ M, // [tile_in_h][tile_in_w][output_channel][num_tiles]
                      float *__restrict__ out, // [batch][output_channel][output_height][output_width]
                      const out_shape_t os,
                      const tiling_info_t ti) {
  int64_t collapsed_dim_size = os.oc * ti.num_tiles;

  typedef float (*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  M_tensor_t M_tensor = (M_tensor_t) M;

  typedef float (*out_tensor_t)[os.oc][os.h][os.w];
  out_tensor_t out_tensor = (out_tensor_t) out;

  #ifdef DEBUG_OUTPUT_PIPELINED
    int64_t start_time = current_time_ns();
  #endif

  #ifdef DEBUG_OUTPUT_PIPELINED
    int64_t get_tile_index_end = current_time_ns();
  #endif

  #pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    int oc   = idx / ti.num_tiles;
    int tile = idx % ti.num_tiles;
    
    float local_M[6][6];
    for (int h = 0; h < 6; h++) {
      for (int w = 0; w < 6; w++) {
        local_M[h][w] = M_tensor[h][w][idx];
      }
    }
    float local_out[4][4];
    output_transform_simd(&local_M[0][0], &local_out[0][0]);
    
    tile_index_t tidx = get_tile_index(tile, ti);
    int batch = tidx.b;
    for (int h = 0; h < ti.tile_out_h; h++) {
      int out_h = tidx.th * ti.tile_out_w + h; 
      if (out_h >= os.h){
        continue;
      }

      for (int w = 0; w < 4; w++) {
        int out_w = tidx.tw * ti.tile_out_w + w;
        if (out_w >= os.w){
          continue;
        }
        out_tensor[batch][oc][out_h][out_w] = local_out[h][w];
      }
    }
  }


  #ifdef DEBUG_OUTPUT_PIPELINED
    int64_t end_time = current_time_ns();
    printf("--->output_pipelined time cost<---\n");
    printf("get_tile_index_time: %ld ns = %ld us = %ld ms\n", 
           get_tile_index_end - start_time, 
           (get_tile_index_end - start_time) / 1000, 
           (get_tile_index_end - start_time) / 1000000);
    printf("output_pipelined_time: %ld ns = %ld us = %ld ms\n", 
           end_time - get_tile_index_end, 
           (end_time - get_tile_index_end) / 1000, 
           (end_time - get_tile_index_end) / 1000000);
    printf("----------------------------------\n");
  #endif
}

void sgemm(const int64_t num_tiles, const int64_t ic, const int64_t oc, float *A, float *B, float *C) {
  typedef float(*A_tensor_t)[ic];
  typedef float(*B_tensor_t)[num_tiles];
  typedef float(*C_tensor_t)[num_tiles];
  A_tensor_t A_tensor = (A_tensor_t)A;
  B_tensor_t B_tensor = (B_tensor_t)B;
  C_tensor_t C_tensor = (C_tensor_t)C;

  // U[oc][ic]
  // V[ic][num_tiles]
  // M[oc][num_tiles]
  memset(C_tensor, 0, sizeof(float) * num_tiles * oc);

  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t i = 0; i < oc; ++i) {
    for (int64_t j = 0; j < ic; ++j) {
      for (int64_t k = 0; k < num_tiles; ++k) {
        C_tensor[i][k] += A_tensor[j][k] * B_tensor[i][j];
      }
    }
  }
}

void V_rev_sgemm(const tiling_info_t ti,
                 const filter_shape_t fs,
                 float * __restrict__ U,  
                 float * __restrict__ V,
                 float*  __restrict__ M) {
  typedef float (*U_tensor_t)[ti.tile_in_w][fs.oc][fs.ic];
  typedef float (*V_tensor_t)[ti.tile_in_w][fs.ic][ti.num_tiles];
  typedef float (*M_tensor_t)[ti.tile_in_w][fs.oc][ti.num_tiles];

  U_tensor_t U_tensor = (U_tensor_t)U;
  V_tensor_t V_tensor = (V_tensor_t)V;
  M_tensor_t M_tensor = (M_tensor_t)M;

  // filter_shape_t: {oc: 64, ic: 3, h: 3, w: 3}
  // tiling_info_t: {bs: 64, num_tile_per_image: 3136, num_tiles: 200704, tiles_on_h: 56, tiles_on_w: 56,
  // tile_in_h: 6, tile_in_w: 6, tile_out_h: 4, tile_out_w: 4}

  // U[tile_in_h][tile_in_w][oc][ic]
  // V[tile_in_h][tile_in_w][ic][num_tiles]
  // M[tile_in_h][tile_in_w][oc][num_tiles]
  // M[tile_in_h][tile_in_w][oc][num_tiles] = \sum_{ic} U[tile_in_h][tile_in_w][oc][ic] * V[tile_in_h][tile_in_w][num_tiles][ic]

  // tile_in_h = 6, tile_in_w = 6
  // num_tiles = ts.tiles_on_h * ts.tiles_on_w  * batch_size
  // fs.h = 3, fs.w = 3
  // ts.tiles_on_h = DIV_UP(os.h, TILE_OUT_H) = DIV_UP(is.h - fs.h + 1, 4)
  // ts.tiles_on_w = DIV_UP(os.w, TILE_OUT_W) = DIV_UP(is.w - fs.w + 1, 4)
  // os.h = is.h - fs.h + 1 = is.h - 2
  // os.w = is.w - fs.w + 1 = is.w - 2
  // -> num_tiles = DIV_UP(is.h - 2, 4) * DIV_UP(is.w - 2, 4) * batch_size
  // -> num_tiles = DIV_UP(input_image_height - filter_height + 1, 4) * 
  //                DIV_UP(input_image_width - filter_width + 1, 4) * 
  //                batch_size
  // num_tiles >>> oc

  memset(M_tensor, 0, sizeof(float) * ti.tile_in_h * ti.tile_in_w * fs.oc * ti.num_tiles);


  int64_t start_time = current_time_ms();
  #pragma omp parallel for collapse(4) schedule(static)
  for (int64_t h = 0; h < ti.tile_in_h; h++) {
    for (int64_t w = 0; w < ti.tile_in_w; w++) {
      // sgemm(ti.num_tiles, fs.ic, fs.oc, &V_tensor[h][w][0][0], &U_tensor[h][w][0][0], &M_tensor[h][w][0][0]);
      // float *base_U = &U_tensor[h][w][0][0];
      for (int64_t oc = 0; oc < fs.oc; oc++) {
        for(int64_t ic = 0; ic < fs.ic; ic++) {
          float *base_M = &M_tensor[h][w][oc][0];
          float *base_V = &V_tensor[h][w][ic][0];
          // float U_val = base_U[oc * fs.ic + ic];
          float U_val = U_tensor[h][w][oc][ic];
          #pragma omp simd
          for(int64_t tile = 0; tile < ti.num_tiles; tile++) {
            base_M[tile] += base_V[tile] * U_val;
          }
        }
      }
    }
  }

  int64_t end_time = current_time_ms();
  printf("tile_in_h X tile_in_w X oc X ic: %ld X %ld X %ld X %ld = %ld\n", 
         ti.tile_in_h, ti.tile_in_w, fs.oc, fs.ic, ti.tile_in_h * ti.tile_in_w * fs.oc * fs.ic);
  printf("test_sgemm time: %ld ms\n", end_time - start_time);
}

void fused_sgemm(const tiling_info_t ti,
                 const filter_shape_t fs,
                 float *U,  
                 float *V,
                 float *M) {
  typedef float (*U_tensor_t)[ti.tile_in_w][fs.oc][fs.ic];
  typedef float (*V_tensor_t)[ti.tile_in_w][ti.num_tiles][fs.ic];
  typedef float (*M_tensor_t)[ti.tile_in_w][fs.oc][ti.num_tiles];
  U_tensor_t U_tensor = (U_tensor_t)U;
  V_tensor_t V_tensor = (V_tensor_t)V;
  M_tensor_t M_tensor = (M_tensor_t)M;
  // filter_shape_t: {oc: 64, ic: 3, h: 3, w: 3}
  // tiling_info_t: {bs: 64, num_tile_per_image: 3136, num_tiles: 200704, tiles_on_h: 56, tiles_on_w: 56,
  // tile_in_h: 6, tile_in_w: 6, tile_out_h: 4, tile_out_w: 4}

  // U[tile_in_h][tile_in_w][oc][ic]
  // V[tile_in_h][tile_in_w][num_tiles][ic]
  // M[tile_in_h][tile_in_w][oc][num_tiles]

  // tile_in_h = 6, tile_in_w = 6
  // num_tiles = ts.tiles_on_h * ts.tiles_on_w  * batch_size
  // fs.h = 3, fs.w = 3
  // ts.tiles_on_h = DIV_UP(os.h, TILE_OUT_H) = DIV_UP(is.h - fs.h + 1, 4)
  // ts.tiles_on_w = DIV_UP(os.w, TILE_OUT_W) = DIV_UP(is.w - fs.w + 1, 4)
  // os.h = is.h - fs.h + 1 = is.h - 2
  // os.w = is.w - fs.w + 1 = is.w - 2
  // -> num_tiles = DIV_UP(is.h - 2, 4) * DIV_UP(is.w - 2, 4) * batch_size
  // -> num_tiles = DIV_UP(input_image_height - 2, 4) * 
  //                DIV_UP(input_image_width - 2, 4) * 
  //                batch_size
  // num_tiles >>> oc

  const int64_t tile_block_size = 64;
  const int64_t oc_block_size = 16;

  int64_t tile_block_end, oc_block_end;
  #pragma omp parallel for collapse(4) schedule(static) private(tile_block_end, oc_block_end)
  for (int64_t h = 0; h < ti.tile_in_h; h++) {
    for (int64_t w = 0; w < ti.tile_in_w; w++) {
      for (int64_t tile_block_start = 0; tile_block_start < ti.num_tiles; tile_block_start += tile_block_size) {
        for (int64_t oc_block_start = 0; oc_block_start < fs.oc; oc_block_start += oc_block_size) {
          tile_block_end = std::min(tile_block_start + tile_block_size, ti.num_tiles);
          oc_block_end = std::min(oc_block_start + oc_block_size, fs.oc);
          float *base_base_U = &U_tensor[h][w][0][0];
          float *base_base_V = &V_tensor[h][w][0][0];
          for (int64_t tile = tile_block_start; tile < tile_block_end; tile++) {
            for (int64_t oc = oc_block_start; oc < oc_block_end; oc++) {
              __m512 vsum = _mm512_setzero_ps();
              float *base_U = &base_base_U[oc * fs.ic];
              float *base_V = &base_base_V[tile * fs.ic];

              int64_t k = 0;
              int64_t k_simd_bound = (fs.ic / 16) * 16;
              for (; k < k_simd_bound; k += 16) {
                __m512 v_val = _mm512_loadu_ps(&base_V[k]);
                __m512 u_val = _mm512_loadu_ps(&base_U[k]);
                vsum = _mm512_fmadd_ps(v_val, u_val, vsum);
              }
              if (k < fs.ic) {
                const int tail = fs.ic - k;
                __mmask16 mask = (1 << tail) - 1;
                __m512 v_val = _mm512_maskz_loadu_ps(mask, &base_V[k]);
                __m512 u_val = _mm512_maskz_loadu_ps(mask, &base_U[k]);
                vsum = _mm512_fmadd_ps(v_val, u_val, vsum);
              }

              float sum = _mm512_reduce_add_ps(vsum);
              M_tensor[h][w][oc][tile] = sum;
            }
          }
        }
      }
    }
  }
}

// 16GB
const size_t default_size = 1L << 36;

typedef struct {
  float *ptr;
  size_t size;
  int in_use;
} memory_manager_t;

memory_manager_t memory_manager = {nullptr, 0, 0};

void *my_simple_reuse_malloc(size_t size) {
  if (memory_manager.in_use == 1) {
    // 并非多线程，所以不需要考虑线程安全
    return nullptr;
  }

  if (memory_manager.ptr == nullptr) {
    // 第一次分配
    // memory_manager.ptr = (float *)malloc(size);
    if (size < default_size){
      size = default_size;
    }
    memory_manager.ptr = (float *)aligned_alloc(32, size);
    memory_manager.size = size;
    memory_manager.in_use = 1;
  }else{
    // 非第一次分配
    if (memory_manager.size < size) {
      // memory_manager.ptr = (float *)realloc(memory_manager.ptr, size);
      // free(memory_manager.ptr);
      memory_manager.ptr = (float *)aligned_alloc(32, size);
      memory_manager.size = size;
    }
  }
  return memory_manager.ptr;
}

void my_simple_reuse_free(void *ptr) {
  if (ptr == memory_manager.ptr) {
    memory_manager.in_use = 0;
  }
}


// 好吧不能改driver.cc，没机会调用了
void my_simple_memory_real_free(void *ptr) {
  if (ptr == memory_manager.ptr) {
    free(memory_manager.ptr);
    memory_manager.ptr = nullptr;
    memory_manager.size = 0;
  }
}


// void fused_winograd_convolution(
//   float *__restrict__ image, // float [batch_num][input_channel_num][image_height][image_width]
//   float *__restrict__ filter, // float [output_channel_num][input_channel_num][FLT_H][FLT_W]
//   float *__restrict__ out, // float [batch_num][output_channel_num][output_height][output_width]
//   const image_shape_t is,
//   const filter_shape_t fs,
//   const out_shape_t os,
//   const tiling_info_t ti) {
//   const U_shape_t us = get_U_shape(fs, ti);
//   const V_shape_t vs = get_V_shape(is, ti);

//   // filter 变换
//   float *total_memory = (float *)my_simple_reuse_malloc(sizeof(float) * (
//     ti.tile_in_h * ti.tile_in_w * us.oc * us.ic +
//     ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic +
//     ti.tile_in_h * ti.tile_in_w * vs.num_tiles * us.oc
//   ));
//   float *U = total_memory;
//   float *V = U + ti.tile_in_h * ti.tile_in_w * us.oc * us.ic;
//   float *M = V + ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic;

//   int64_t num_tiles = ti.num_tiles;

//   for (int64_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
   
    
//   }
// }


void winograd_convolution(
    float *__restrict__ image, /** float [batch_num][input_channel_num][image_height][image_width] */
    const int image_height,
    const int image_width,
    const int input_channel_num,
    float *__restrict__ filter, /** float [output_channel_num][input_channel_num][FLT_H][FLT_W] */
    const int output_channel_num,
    const int batch_num,
    float *__restrict__ out) {

  const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
  const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
  const out_shape_t os = get_output_shape(is, fs);
  const tiling_info_t ti = get_tiling_info(is, os);

  // CONSTS: 
  // ts.tile_in_h = TILE_IN_H; = 6
  // ts.tile_in_w = TILE_IN_W; = 6
  // ts.tile_out_h = TILE_OUT_H; = 4
  // ts.tile_out_w = TILE_OUT_W; = 4
  const U_shape_t us = get_U_shape(fs, ti);
  const V_shape_t vs = get_V_shape(is, ti);

  #ifdef DEBUG
    int64_t start_time = current_time_ms();
  #endif

  // float *total_memory = (float *)malloc(sizeof(float) * (
  //   ti.tile_in_h * ti.tile_in_w * us.oc * us.ic +
  //   ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic +
  //   ti.tile_in_h * ti.tile_in_w * vs.num_tiles * us.oc
  // ));
  float *total_memory = (float *)my_simple_reuse_malloc(sizeof(float) * (
    ti.tile_in_h * ti.tile_in_w * us.oc * us.ic +
    ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic +
    ti.tile_in_h * ti.tile_in_w * vs.num_tiles * us.oc
  ));
  float *U = total_memory;
  float *V = U + ti.tile_in_h * ti.tile_in_w * us.oc * us.ic;
  float *M = V + ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic;

  // float *U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
  // float *V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
  // float *M = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * us.oc);

  #ifdef DEBUG
    printf("----Params----\n");
    printf("image_shape_t: {bs: %ld, ic: %ld, h: %ld, w: %ld}\n", is.bs, is.ic, is.h, is.w);
    printf("filter_shape_t: {oc: %ld, ic: %ld, h: %ld, w: %ld}\n", fs.oc, fs.ic, fs.h, fs.w);
    printf("out_shape_t: {bs: %ld, oc: %ld, h: %ld, w: %ld}\n", os.bs, os.oc, os.h, os.w);
    printf("tiling_info_t: {bs: %ld, num_tile_per_image: %ld, num_tiles: %ld, tiles_on_h: %ld, tiles_on_w: %ld,\n", 
           ti.bs, ti.num_tile_per_image, ti.num_tiles, ti.tiles_on_h, ti.tiles_on_w);
    printf("tile_in_h: %ld, tile_in_w: %ld, tile_out_h: %ld, tile_out_w: %ld\n", 
           ti.tile_in_h, ti.tile_in_w, ti.tile_out_h, ti.tile_out_w);
    printf("U_shape_t: {oc: %ld, ic: %ld}\n", us.oc, us.ic);
    printf("V_shape_t: {num_tiles: %ld, ic: %ld}\n", vs.num_tiles, vs.ic);
    printf("U_tensor size: %ld bytes\n", sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
    printf("V_tensor size: %ld bytes\n", sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
    printf("M_tensor size: %ld bytes\n", sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * us.oc);
    printf("total_memory size: %ld bytes\n", sizeof(float) * (
      ti.tile_in_h * ti.tile_in_w * us.oc * us.ic +
      ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic +
      ti.tile_in_h * ti.tile_in_w * vs.num_tiles * us.oc
    ));
  #endif

  #ifdef DEBUG
    int64_t malloc_time = current_time_ms();
  #endif

  // float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
  // filter_packing(filter, packed_filter, fs);
  #ifdef DEBUG
    int64_t filter_packing_time = current_time_ms();
  #endif

  // filter_transform(packed_filter, U, fs, us, us.oc * us.ic);
  filter_transform(filter, U, fs, us, us.oc * us.ic);
  #ifdef DEBUG
    int64_t filter_transform_time = current_time_ms();
  #endif

  // float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
  // image_packing(image, packed_image, is, ti);
  #ifdef DEBUG
    int64_t image_packing_time = current_time_ms();
  #endif

  // image_transform(packed_image, V, vs, ti, vs.ic * vs.num_tiles);
  image_transform(image, V, is, vs, ti);

  // V_rev: 交换最后两个维度
  // float *V_rev = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
  // image_transform_reverse(image, V, is, vs, ti);
  // check if v == v_rev
  // V[tile_in_h][tile_in_w][num_tiles][ic]
  // V_rev[tile_in_h][tile_in_w][ic][num_tiles]
  // for (int64_t i = 0; i < ti.tile_in_h; i++) {
  //   for (int64_t j = 0; j < ti.tile_in_w; j++) {
  //     float *base_v = &V[i * ti.tile_in_w * vs.num_tiles * vs.ic + j * vs.num_tiles * vs.ic];
  //     float *base_v_rev = &V_rev[i * ti.tile_in_w * vs.ic * vs.num_tiles + j * vs.ic * vs.num_tiles];
  //     for (int64_t k = 0; k < vs.num_tiles; k++) {
  //       for (int64_t l = 0; l < vs.ic; l++) {
  //         int64_t v_pos = k * vs.ic + l;
  //         int64_t v_rev_pos = l * vs.num_tiles + k;
  //         // if (base_v[v_pos] != base_v_rev[v_rev_pos]) {
  //         //   printf("V[%ld][%ld][%ld][%ld] = %f, V_rev[%ld][%ld][%ld][%ld] = %f\n", i, j, k, l, V[v_pos], i, j, l, k, V_rev[v_rev_pos]);
  //         // }
  //         base_v[v_pos] = base_v_rev[v_rev_pos];
  //       }
  //     }
  //   }
  // }

  #ifdef DEBUG
    int64_t image_transform_time = current_time_ms();  
  #endif

  // fused sgemm
  fused_sgemm(ti, fs, U, V, M);
  // cuda_fused_sgemm(ti, fs, U, V, M);
  
  // float *M2 = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
  // V_rev_sgemm(ti, fs, U, V, M);


  // check if M == M2
  // for (int64_t i = 0; i < ti.tile_in_h; i++) {
  //   for (int64_t j = 0; j < ti.tile_in_w; j++) {
  //     // 获取M和M2在当前位置(i,j)的基址指针，用于比较两个矩阵的值
  //     float *base_m = &M[i * ti.tile_in_w * us.oc * vs.num_tiles + j * us.oc * vs.num_tiles];
  //     float *base_m2 = &M2[i * ti.tile_in_w * us.oc * vs.num_tiles + j * us.oc * vs.num_tiles];
  //     // 遍历输出通道维度
  //     for (int64_t k = 0; k < us.oc; k++) {
  //       for (int64_t l = 0; l < vs.num_tiles; l++) {
  //         int64_t offset = k * vs.num_tiles + l;
  //         float diff = std::abs(base_m[offset] - base_m2[offset]);
  //         float diff_percent = diff / std::abs(base_m[offset]);
  //         if (diff_percent > 1e-2) {
  //           printf("M[%ld][%ld][%ld][%ld] = %f, M2[%ld][%ld][%ld][%ld] = %f\n",
  //             i, j, k, l, base_m[offset],
  //             i, j, k, l, base_m2[offset]);
  //         }
  //       }
  //     }
  //   }
  // }

  #ifdef DEBUG
    int64_t sgemm_time = current_time_ms();
  #endif


  // output_transform_locality(M, Y, ti, us.oc * vs.num_tiles);
  #ifdef DEBUG
    int64_t output_transform_time = current_time_ms();
  #endif

  // output_unpacking_store_locality(Y, out, os, ti);

  
  output_transform(M, out, os, ti);
  #ifdef DEBUG
    int64_t output_unpacking_store_time = current_time_ms();
  #endif


  // free(packed_filter);
  // free(packed_image);
  // free(U);
  // free(V);
  // free(M);
  // free(total_memory);
  // free(Y);
  my_simple_reuse_free(total_memory);

  #ifdef DEBUG
    int64_t free_time = current_time_ms();
  #endif

  #ifdef DEBUG
    printf("---time cost---\n");
    printf("malloc_time: %ld, delta: %ldms\n", malloc_time, malloc_time - start_time);
    printf("filter_packing_time: %ld, delta: %ldms\n", filter_packing_time, filter_packing_time - malloc_time);
    printf("filter_transform_time: %ld, delta: %ldms\n", filter_transform_time, filter_transform_time - filter_packing_time);
    printf("image_packing_time: %ld, delta: %ldms\n", image_packing_time, image_packing_time - filter_transform_time);
    printf("image_transform_time: %ld, delta: %ldms\n", image_transform_time, image_transform_time - image_packing_time);
    printf("num of sgemm times: %ld, h: %ld, w: %ld\n", ti.tile_in_h * ti.tile_in_w, ti.tile_in_h, ti.tile_in_w);
    printf("M(num_tiles) x N(oc) x K(ic): %ld x %ld x %ld\n", ti.num_tiles, us.oc, us.ic);
    printf("sgemm_time: %ld, delta: %ldms\n", sgemm_time, sgemm_time - image_transform_time);
    printf("output_transform_time: %ld, delta: %ldms\n", output_transform_time, output_transform_time - sgemm_time);
    printf("output_unpacking_store_time: %ld, delta: %ldms\n", output_unpacking_store_time, output_unpacking_store_time - output_transform_time);
    printf("free_time: %ld, delta: %ldms\n", free_time, free_time - output_unpacking_store_time);
    printf("--------------------------------\n");
  #endif
}