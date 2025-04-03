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
#include "memory.cpp"
#include "flag.h"

static const float G[6][3] = {
  {1/4.f, 0.f, 0.f},
  {-1/6.f, -1/6.f, -1/6.f},
  {-1/6.f, 1/6.f, -1/6.f},
  {1/24.f, 1/12.f, 1/6.f},
  {1/24.f, -1/12.f, 1/6.f},
  {0.f, 0.f, 1.f},
};
static const float GT[3][6] = {
  {1/4.f, -1/6.f, -1/6.f, 1/24.f, 1/24.f, 0.f},
  {0.f, -1/6.f, 1/6.f, 1/12.f, -1/12.f, 0.f},
  {0.f, -1/6.f, -1/6.f, 1/6.f, 1/6.f, 1.f},
};

inline void filter_transform_simd_4x3(float filter[3][3],
                                      float output[6][6]) {
    static const __m128 v1_4      = _mm_set1_ps(1.0f / 4.0f);
    static const __m128 v_neg1_6  = _mm_set1_ps(-1.0f / 6.0f);
    static const __m128 v1_6      = _mm_set1_ps(1.0f / 6.0f);
    static const __m128 v1_24     = _mm_set1_ps(1.0f / 24.0f);
    static const __m128 v1_12     = _mm_set1_ps(1.0f / 12.0f);
    static const __m128 v_neg1_12 = _mm_set1_ps(-1.0f / 12.0f);
    __m128 g0 = _mm_setr_ps(filter[0][0],
                            filter[0][1],
                            filter[0][2],
                            0.0f);
    __m128 g1 = _mm_setr_ps(filter[1][0],
                            filter[1][1],
                            filter[1][2],
                            0.0f);
    __m128 g2 = _mm_setr_ps(filter[2][0],
                            filter[2][1],
                            filter[2][2],
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

    transform_row(U0, output[0]);
    transform_row(U1, output[1]);
    transform_row(U2, output[2]);
    transform_row(U3, output[3]);
    transform_row(U4, output[4]);
    transform_row(U5, output[5]);
}


void filter_transform(float *__restrict__ filter,
                      float *__restrict__ V,
                      const filter_shape_t fs,
                      const U_shape_t us) {
  typedef float (*filter_tensor_t)[fs.ic][fs.h][fs.w];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  // typedef float (*V_tensor_t)[us.w][collapsed_dim_size];
  typedef float (*V_tensor_t)[us.w][us.oc][us.ic];
  V_tensor_t V_tensor = (V_tensor_t)V;


  #pragma omp parallel for collapse(2) schedule(static)
  // for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
  for (int64_t oc_idx = 0; oc_idx < us.oc; oc_idx++) {
    for (int64_t ic_idx = 0; ic_idx < us.ic; ic_idx++) {
      // int oc     = (int)(idx / fs.ic);
      // int ic_idx = (int)(idx % fs.ic);
      // float *filter_tensor_base = &filter_tensor[oc_idx][ic_idx][0][0];
    
      float output_local[6][6];
      filter_transform_simd_4x3((float (*)[3])&filter_tensor[oc_idx][ic_idx][0][0], output_local);
    
      for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
          V_tensor[i][j][oc_idx][ic_idx] = output_local[i][j];
        }
      }
    }
  }
}

const static float B[6][6] = {
    {4.f, 0.f, -5.f, 0.f, 1.f, 0.f},
    {0.f, -4.f, -4.f, 1.f, 1.f, 0.f},
    {0.f, 4.f, -4.f, -1.f, 1.f, 0.f},
    {0.f, -2.f, -1.f, 2.f, 1.f, 0.f},
    {0.f, 2.f, -1.f, -2.f, 1.f, 0.f},
    {0.f, 4.f, 0.f, -5.f, 0.f, 1.f},
};

const static float B_T[6][6] = {
    {4.f, 0.f, 0.f, 0.f, 0.f, 0.f},
    {0.f, -4.f, 4.f, 0.f, 0.f, 0.f},
    {-5.f, -4.f, -4.f, -2.f, 2.f, 4.f},
    {0.f, 1.f, -1.f, 2.f, -2.f, -5.f},
    {1.f, 1.f, 1.f, 1.f, 1.f, 0.f},
    {0.f, 0.f, 0.f, 0.f, 0.f, 1.f},
};
// output = B^T * input * B
inline void image_transform_simd_4x3(float input_d[6][6],
                                 float output_d[6][6]) {
    float temp[6][6];

    static const __m128 v_4 = _mm_set1_ps(4.f);
    static const __m128 v_neg5 = _mm_set1_ps(-5.f);
    static const __m128 v_neg4 = _mm_set1_ps(-4.f);
    static const __m128 v_neg1 = _mm_set1_ps(-1.f);
    static const __m128 v_2 = _mm_set1_ps(2.f);
    static const __m128 v_neg2 = _mm_set1_ps(-2.f);
    
    
    __m128 r0 = _mm_loadu_ps(&input_d[0][0]); // r0 = [d00, d01, d02, d03]
    __m128 r1 = _mm_loadu_ps(&input_d[1][0]); // r1 = [d10, d11, d12, d13]
    __m128 r2 = _mm_loadu_ps(&input_d[2][0]); // r2 = [d20, d21, d22, d23]
    __m128 r3 = _mm_loadu_ps(&input_d[3][0]); // r3 = [d30, d31, d32, d33]
    __m128 r4 = _mm_loadu_ps(&input_d[4][0]); // r4 = [d40, d41, d42, d43]  
    __m128 r5 = _mm_loadu_ps(&input_d[5][0]); // r5 = [d50, d51, d52, d53]

    __m128 v0 = _mm_add_ps(
                    _mm_mul_ps(v_4, r0),
                    _mm_add_ps(_mm_mul_ps(v_neg5, r2), r4)
                  ); // v0 = 4 * r0 + 0 * r1 + 0 * r2 + 0 * r3 + 0 * r4 + 0 * r5
    _mm_storeu_ps(&temp[0][0], v0);

    __m128 v1 = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(v_neg4, r1),
                                _mm_mul_ps(v_neg4, r2)),
                    _mm_add_ps(r3, r4)
                  );
    _mm_storeu_ps(&temp[1][0], v1);

    __m128 v2 = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(v_4, r1),
                                _mm_mul_ps(v_neg4, r2)),
                    _mm_add_ps(_mm_mul_ps(v_neg1, r3), r4)
                  );
    _mm_storeu_ps(&temp[2][0], v2);

    __m128 v3 = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(v_neg2, r1),
                                _mm_mul_ps(v_neg1, r2)),
                    _mm_add_ps(_mm_mul_ps(v_2, r3), r4)
                  );
    _mm_storeu_ps(&temp[3][0], v3);

    __m128 v4 = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(v_2, r1),
                                _mm_mul_ps(v_neg1, r2)),
                    _mm_add_ps(_mm_mul_ps(v_neg2, r3), r4)
                  );
    _mm_storeu_ps(&temp[4][0], v4);

    __m128 v5 = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(v_4, r1),
                                _mm_mul_ps(v_neg5, r3)),
                    r5
                  );
    _mm_storeu_ps(&temp[5][0], v5);
    
    // 处理剩余的2个元素
    auto transform_row = [&](int w) {
        float a0 = input_d[0][w];
        float a1 = input_d[1][w];
        float a2 = input_d[2][w];
        float a3 = input_d[3][w];
        float a4 = input_d[4][w];
        float a5 = input_d[5][w];
        temp[0][w] = 4.f * a0 + (-5.f) * a2 + a4;
        temp[1][w] = -4.f * a1 + (-4.f) * a2 + a3 + a4;
        temp[2][w] = 4.f * a1 + (-4.f) * a2 + (-1.f)*a3 + a4;
        temp[3][w] = -2.f * a1 + (-1.f)*a2 + 2.f * a3 + a4;
        temp[4][w] = 2.f * a1 + (-1.f)*a2 + (-2.f)*a3 + a4;
        temp[5][w] = 4.f * a1 + (-5.f)*a3 + a5;
    };
    transform_row(4);
    transform_row(5);

    auto transform_column = [&](int h) {
      __m128 vec = _mm_loadu_ps(&temp[h][0]);
        float v4 = temp[h][4];
        float v5 = temp[h][5];

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

        output_d[h][0] = t0;
        output_d[h][1] = t1;
        output_d[h][2] = t2;
        output_d[h][3] = t3;
        output_d[h][4] = t4;
        output_d[h][5] = t5;
    };

    transform_column(0);
    transform_column(1);
    transform_column(2);
    transform_column(3);
    transform_column(4);
    transform_column(5);
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
      image_transform_simd_4x3(local_tile, final_tile);

      // int64_t idx = ic_idx * ti.num_tiles + tile_idx;
      int64_t idx = tile_idx * is.ic + ic_idx;
      for (int h = 0; h < ti.tile_in_h; h++) {
        for (int w = 0; w < ti.tile_in_w; w++) {
          V_tensor[h][w][idx] = final_tile[h][w];
        }
      }
    }
  }
}

// 固定tile：
// ts.tile_in_h = TILE_IN_H; = 6
// ts.tile_in_w = TILE_IN_W; = 6
// ts.tile_out_h = TILE_OUT_H; = 4
// ts.tile_out_w = TILE_OUT_W; = 4

// F(4,3)
static const float A[4][6] = {
    {1.f,  1.f,  1.f,  1.f,  1.f,  0.f},
    {0.f,  1.f, -1.f,  2.f, -2.f,  0.f},
    {0.f,  1.f,  1.f,  4.f,  4.f,  0.f},
    {0.f,  1.f, -1.f,  8.f, -8.f,  1.f}
};
static const float AT[6][4] = {
    {1.f,  0.f,  0.f,  0.f},
    {1.f,  1.f,  1.f,  1.f},
    {1.f, -1.f,  1.f, -1.f},
    {1.f,  2.f,  4.f,  8.f},
    {1.f, -2.f,  4.f, -8.f},
    {0.f,  0.f,  0.f,  1.f},
};
// Y = A * M * A^T
inline void output_transform_simd_4x3(const float input_m[6][6] , // [tile_in_h=6][tile_in_w=6]
                                  float output[4][4]) {     // [output_tile_h=4][output_tile_w=4]
    float tempH[4][6] = {0};

    const __m128 v_1 = _mm_set1_ps(1.f);
    const __m128 v_neg1 = _mm_set1_ps(-1.f);
    const __m128 v_2 = _mm_set1_ps(2.f);
    const __m128 v_neg2 = _mm_set1_ps(-2.f);
    const __m128 v_4 = _mm_set1_ps(4.f);
    const __m128 v_8 = _mm_set1_ps(8.f);
    const __m128 v_neg8 = _mm_set1_ps(-8.f);
    
    for (int j = 0; j < 4; j += 4) {
        __m128 t0 = _mm_setzero_ps();
        __m128 t1 = _mm_setzero_ps();
        __m128 t2 = _mm_setzero_ps();
        __m128 t3 = _mm_setzero_ps();
        // k = 0
        {
            __m128 m_col0 = _mm_loadu_ps(&input_m[0][j]);
            t0 = _mm_fmadd_ps(m_col0, v_1, t0);
            // t1 = _mm_fmadd_ps(m_col0, v_0, t1);
            // t2 = _mm_fmadd_ps(m_col0, v_0, t2);
            // t3 = _mm_fmadd_ps(m_col0, v_0, t3);
        }
        // k = 1
        {
            __m128 m = _mm_loadu_ps(&input_m[1][j]);
            t0 = _mm_fmadd_ps(m, v_1, t0);
            t1 = _mm_fmadd_ps(m, v_1, t1);
            t2 = _mm_fmadd_ps(m, v_1, t2);
            t3 = _mm_fmadd_ps(m, v_1, t3);
        }
        // k = 2
        {
            __m128 m = _mm_loadu_ps(&input_m[2][j]);
            t0 = _mm_fmadd_ps(m, v_1, t0);
            t1 = _mm_fmadd_ps(m, v_neg1, t1);
            t2 = _mm_fmadd_ps(m, v_1, t2);
            t3 = _mm_fmadd_ps(m, v_neg1, t3);
        }
        // k = 3
        {
            __m128 m = _mm_loadu_ps(&input_m[3][j]);
            t0 = _mm_fmadd_ps(m, v_1, t0);
            t1 = _mm_fmadd_ps(m, v_2, t1);
            t2 = _mm_fmadd_ps(m, v_4, t2);
            t3 = _mm_fmadd_ps(m, v_8, t3);
        }
        // k = 4
        {
            __m128 m = _mm_loadu_ps(&input_m[4][j]);
            t0 = _mm_fmadd_ps(m, v_1, t0);
            t1 = _mm_fmadd_ps(m, v_neg2, t1);
            t2 = _mm_fmadd_ps(m, v_4, t2);
            t3 = _mm_fmadd_ps(m, v_neg8, t3);
        }
        // k = 5
        {
            __m128 m = _mm_loadu_ps(&input_m[5][j]);
            // t0 = _mm_fmadd_ps(m, v_0, t0);
            // t1 = _mm_fmadd_ps(m, v_0, t1);
            // t2 = _mm_fmadd_ps(m, v_0, t2);
            t3 = _mm_fmadd_ps(m, v_1, t3);
        }
        _mm_storeu_ps(&tempH[0][j], t0);
        _mm_storeu_ps(&tempH[1][j], t1);
        _mm_storeu_ps(&tempH[2][j], t2);
        _mm_storeu_ps(&tempH[3][j], t3);
    }
    for (int j = 4; j < 6; j++) {
        for (int k = 0; k < 6; k++) {
            tempH[0][j] += input_m[k][j] * A[0][k];
            tempH[1][j] += input_m[k][j] * A[1][k];
            tempH[2][j] += input_m[k][j] * A[2][k];
            tempH[3][j] += input_m[k][j] * A[3][k];
        }
    }

    static const __m128 v_col0 = _mm_set_ps(AT[3][0], AT[2][0], AT[1][0], AT[0][0]);
    static const __m128 v_col1 = _mm_set_ps(AT[3][1], AT[2][1], AT[1][1], AT[0][1]);
    static const __m128 v_col2 = _mm_set_ps(AT[3][2], AT[2][2], AT[1][2], AT[0][2]);
    static const __m128 v_col3 = _mm_set_ps(AT[3][3], AT[2][3], AT[1][3], AT[0][3]);
    auto transform_row = [&](int i) {
        __m128 t = _mm_loadu_ps(&tempH[i][0]);
        float sum0 = _mm_cvtss_f32(_mm_dp_ps(t, v_col0, 0xF1));
        float sum1 = _mm_cvtss_f32(_mm_dp_ps(t, v_col1, 0xF1));
        float sum2 = _mm_cvtss_f32(_mm_dp_ps(t, v_col2, 0xF1));
        float sum3 = _mm_cvtss_f32(_mm_dp_ps(t, v_col3, 0xF1));

        // sum0 += tempH[i][4] * AT[4][0] + tempH[i][5] * AT[5][0];
        // sum1 += tempH[i][4] * AT[4][1] + tempH[i][5] * AT[5][1];
        // sum2 += tempH[i][4] * AT[4][2] + tempH[i][5] * AT[5][2];
        // sum3 += tempH[i][4] * AT[4][3] + tempH[i][5] * AT[5][3];

        sum0 += tempH[i][4];
        sum1 += tempH[i][4] * -2;
        sum2 += tempH[i][4] * 4;
        sum3 += tempH[i][4] * -8 + tempH[i][5];

        __m128 result = _mm_setr_ps(sum0, sum1, sum2, sum3);
        _mm_storeu_ps(&output[i][0], result);
    };

    transform_row(0);
    transform_row(1);
    transform_row(2);
    transform_row(3);
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
    output_transform_simd_4x3(local_M, local_out);
    
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
  const int64_t k_simd_bound = (fs.ic / 16) * 16;
  const int64_t k_simd_iter_num = k_simd_bound / 16;
  const int64_t k_simd_bound_tail = fs.ic - (k_simd_iter_num * 16);

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
                            
              for (int64_t i = 0; i < k_simd_iter_num; i++) {
                __m512 v_val = _mm512_loadu_ps(&base_V[k]);
                __m512 u_val = _mm512_loadu_ps(&base_U[k]);
                vsum = _mm512_fmadd_ps(v_val, u_val, vsum);
                k += 16;
              }

              // 尾部不足16的处理
              if (k < fs.ic) {
                __mmask16 mask = (1 << k_simd_bound_tail) - 1;
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

  // U 和 V 最后一维都是is.ic，但是不足16的倍数，padding到16
  int64_t padding_alloc_ic = DIV_UP(is.ic, 16) * 16;

  // 分配内存
  float *total_memory = (float *)my_simple_reuse_malloc(sizeof(float) * (
    ti.tile_in_h * ti.tile_in_w * us.oc * padding_alloc_ic +
    ti.tile_in_h * ti.tile_in_w * vs.num_tiles * padding_alloc_ic +
    ti.tile_in_h * ti.tile_in_w * vs.num_tiles * us.oc
  ));
  float *U = total_memory;

  // 让V和M对齐32字节
  int64_t U_offset_32 = DIV_UP(ti.tile_in_h * ti.tile_in_w * us.oc * padding_alloc_ic, 32) * 32;
  float *V = U + U_offset_32;
  int64_t V_offset_32 = DIV_UP(ti.tile_in_h * ti.tile_in_w * vs.num_tiles * padding_alloc_ic, 32) * 32;
  float *M = V + V_offset_32;

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
  filter_transform(filter, U, fs, us);
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