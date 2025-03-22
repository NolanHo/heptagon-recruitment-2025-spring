#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include "utils.h"
#include <time.h>
#include <omp.h>

#define DEBUG

int64_t current_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)(ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL);
}

void image_transform(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size) {
  typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  V_tensor_t V_tensor = (V_tensor_t)V;

  // #ifdef DEBUG
  //   printf("[image_transform], v_shape_t.h: %ld, v_shape_t.w: %ld, v_shape_t.num_tiles: %ld, v_shape_t.ic: %ld\n", vs.h, vs.w, vs.num_tiles, vs.ic);
  //   printf("[image_transform], tiling_info_t.tile_in_h: %ld, tiling_info_t.tile_in_w: %ld, tiling_info_t.num_tiles: %ld, tiling_info_t.tiles_on_h: %ld, tiling_info_t.tiles_on_w: %ld\n", ti.tile_in_h, ti.tile_in_w, ti.num_tiles, ti.tiles_on_h, ti.tiles_on_w);
  //   printf("[image_transform], tiling_info_t.bs: %ld, tiling_info_t.num_tile_per_image: %ld, tiling_info_t.tiles_on_h: %ld, tiling_info_t.tiles_on_w: %ld\n", ti.bs, ti.num_tile_per_image, ti.tiles_on_h, ti.tiles_on_w);
  //   printf("[image_transform], collapsed_dim_size: %ld\n", collapsed_dim_size);
  // #endif

  #pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    #pragma omp simd
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      float z0, z1, z2, z3, z4, z5, z6;
      z6 = packed_image_tensor[0][w][idx];

      z0 = 4.0f * z6;

      z6 = packed_image_tensor[1][w][idx];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = packed_image_tensor[2][w][idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = packed_image_tensor[3][w][idx];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = packed_image_tensor[4][w][idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = packed_image_tensor[5][w][idx];

      z5 += z6;

      V_tensor[0][w][idx] = z0;
      V_tensor[1][w][idx] = z1;
      V_tensor[2][w][idx] = z2;
      V_tensor[3][w][idx] = z3;
      V_tensor[4][w][idx] = z4;
      V_tensor[5][w][idx] = z5;
    }

    #pragma omp simd  
    for (int64_t h = 0; h < ti.tile_in_h; ++h) {
      float z0, z1, z2, z3, z4, z5, z6;
      z6 = V_tensor[h][0][idx];

      z0 = 4.0f * z6;

      z6 = V_tensor[h][1][idx];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = V_tensor[h][2][idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = V_tensor[h][3][idx];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = V_tensor[h][4][idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = V_tensor[h][5][idx];

      z5 += z6;

      V_tensor[h][0][idx] = z0;
      V_tensor[h][1][idx] = z1;
      V_tensor[h][2][idx] = z2;
      V_tensor[h][3][idx] = z3;
      V_tensor[h][4][idx] = z4;
      V_tensor[h][5][idx] = z5;
    }
  }
}

void filter_transform(float *__restrict__ packed_filter,
                      float *__restrict__ U,
                      const filter_shape_t fs,
                      const U_shape_t us,
                      const int64_t collapsed_dim_size) {
  typedef float(*packed_filter_tensor_t)[fs.w][collapsed_dim_size];
  typedef float(*U_tensor_t)[us.w][collapsed_dim_size];
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
  U_tensor_t U_tensor = (U_tensor_t)U;

  #pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    #pragma omp simd
    for (int64_t w = 0; w < fs.w; ++w) {
      float z0, z1, z2, z3, z4, z5, z6;
      z6 = packed_filter_tensor[0][w][idx];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = packed_filter_tensor[1][w][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = packed_filter_tensor[2][w][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[0][w][idx] = z0;
      U_tensor[1][w][idx] = z1;
      U_tensor[2][w][idx] = z2;
      U_tensor[3][w][idx] = z3;
      U_tensor[4][w][idx] = z4;
      U_tensor[5][w][idx] = z5;
    }

    #pragma omp simd
    for (int64_t h = 0; h < us.h; ++h) {
      float z0, z1, z2, z3, z4, z5, z6;
      z6 = U_tensor[h][0][idx];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = U_tensor[h][1][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = U_tensor[h][2][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[h][0][idx] = z0;
      U_tensor[h][1][idx] = z1;
      U_tensor[h][2][idx] = z2;
      U_tensor[h][3][idx] = z3;
      U_tensor[h][4][idx] = z4;
      U_tensor[h][5][idx] = z5;
    }
  }
}

void output_transform(float *__restrict__ M,
                      float *__restrict__ Y,
                      const tiling_info_t ti,
                      const int64_t collapsed_dim_size) {
  typedef float(*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*Y_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  M_tensor_t M_tensor = (M_tensor_t)M;
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;

  #pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {

    #pragma omp simd
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      float z0, z1, z2, z3, z4;
      z4 = M_tensor[0][w][idx];
      z0 = z4;

      z4 = M_tensor[1][w][idx];
      z0 = z0 + z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = M_tensor[2][w][idx];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = M_tensor[3][w][idx];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = M_tensor[4][w][idx];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = M_tensor[5][w][idx];
      z3 += z4;

      Y_tensor[0][w][idx] = z0;
      Y_tensor[1][w][idx] = z1;
      Y_tensor[2][w][idx] = z2;
      Y_tensor[3][w][idx] = z3;
    }

    #pragma omp simd
    for (int64_t h = 0; h < ti.tile_out_h; ++h) {
      float z0, z1, z2, z3, z4;
      z4 = Y_tensor[h][0][idx];

      z0 = z4;

      z4 = Y_tensor[h][1][idx];
      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = Y_tensor[h][2][idx];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = Y_tensor[h][3][idx];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = Y_tensor[h][4][idx];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = Y_tensor[h][5][idx];

      z3 += z4;

      Y_tensor[h][0][idx] = z0;
      Y_tensor[h][1][idx] = z1;
      Y_tensor[h][2][idx] = z2;
      Y_tensor[h][3][idx] = z3;
    }
  }
}

#define FILTER_PACKING_SMALL_THRESHOLD 1887436818874368

void filter_packing(float *__restrict__ filter, float *__restrict__ packed_filter, const filter_shape_t fs) {
  typedef float(*filter_tensor_t)[fs.ic][fs.h][fs.w];
  typedef float(*packed_filter_tensor_t)[fs.w][fs.oc][fs.ic];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;

  // for (int64_t h = 0; h < fs.h; ++h)
  //   for (int64_t w = 0; w < fs.w; ++w)
  //     for (int64_t oc = 0; oc < fs.oc; oc++)
  //       for (int64_t ic = 0; ic < fs.ic; ic++)
  //         packed_filter_tensor[h][w][oc][ic] = filter_tensor[oc][ic][h][w];

  // too small, no omp
  #pragma omp parallel for collapse(3) schedule(static)
  for (int64_t h = 0; h < fs.h; ++h)
    for (int64_t w = 0; w < fs.w; ++w)
      for (int64_t oc = 0; oc < fs.oc; oc++)
        #pragma omp simd
        for (int64_t ic = 0; ic < fs.ic; ic++)
          packed_filter_tensor[h][w][oc][ic] = filter_tensor[oc][ic][h][w];
}

void image_packing(float *__restrict__ image,
                   float *__restrict__ packed_image,
                   const image_shape_t is,
                   const tiling_info_t ti) {
  typedef float(*packedImage_tensor_t)[ti.tile_in_w][ti.num_tiles][is.ic];
  typedef float(*image_tensor_t)[is.ic][is.h][is.w];

  packedImage_tensor_t packed_image_tensor = (packedImage_tensor_t)packed_image;
  image_tensor_t image_tensor = (image_tensor_t)image;

  // for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
  //   for (int64_t ic = 0; ic < is.ic; ic++) {
  //     for (int64_t h = 0; h < ti.tile_in_h; ++h) {
  //       for (int64_t w = 0; w < ti.tile_in_w; ++w) {
  //         tile_index_t tidx = get_tile_index(tile, ti);
  //         int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
  //         if (hh * 4 + h < is.h && ww * 4 + w < is.w)
  //           packed_image_tensor[h][w][tile][ic] = image_tensor[batch][ic][(hh * 4 + h)][(ww * 4 + w)];
  //         else
  //           packed_image_tensor[h][w][tile][ic] = 0;
  //       }
  //     }
  //   }
  // }

  #pragma omp parallel for schedule(static)
  for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
    tile_index_t tidx = get_tile_index(tile, ti);
    int64_t batch = tidx.b;
    int64_t base_h = tidx.th * 4; 
    int64_t base_w = tidx.tw * 4;

    for (int64_t ic = 0; ic < is.ic; ic++) {
      for (int64_t h = 0; h < ti.tile_in_h; h++) {
        #pragma omp simd
        for (int64_t w = 0; w < ti.tile_in_w; w++) {
          int64_t img_h = base_h + h;
          int64_t img_w = base_w + w;
          if (img_h < is.h && img_w < is.w)
            packed_image_tensor[h][w][tile][ic] = image_tensor[batch][ic][img_h][img_w];
          else
            packed_image_tensor[h][w][tile][ic] = 0;
        }
      }
    }
  }
}

void output_unpacking_store(float *__restrict__ Y,
                            float *__restrict__ out,
                            const out_shape_t os,
                            const tiling_info_t ti) {
  typedef float(*Y_tensor_t)[ti.tile_in_w][os.oc][ti.num_tiles];
  typedef float(*out_tensor_t)[os.oc][os.h][os.w];
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  out_tensor_t out_tensor = (out_tensor_t)out;

  // for (int64_t h = 0; h < ti.tile_out_h; ++h) {
  //   for (int64_t w = 0; w < ti.tile_out_w; ++w) {
  //     for (int64_t oc = 0; oc < os.oc; oc++) {
  //       for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
  //         tile_index_t tidx = get_tile_index(tile, ti);
  //         int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
  //         if (hh * 4 + h < os.h && ww * 4 + w < os.w)
  //           out_tensor[batch][oc][(hh * 4 + h)][(ww * 4 + w)] = Y_tensor[h][w][oc][tile];
  //       }
  //     }
  //   }
  // }

  #pragma omp parallel for collapse(3) schedule(static)
  for (int64_t h = 0; h < ti.tile_out_h; ++h) {
    for (int64_t w = 0; w < ti.tile_out_w; ++w) {
      for (int64_t oc = 0; oc < os.oc; ++oc) {
        #pragma omp simd
        for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
          tile_index_t tidx = get_tile_index(tile, ti);
          int64_t batch = tidx.b;
          int64_t ww    = tidx.tw;
          int64_t hh    = tidx.th;

          int64_t out_h = hh * 4 + h;
          int64_t out_w = ww * 4 + w;
          if (out_h < os.h && out_w < os.w)
            out_tensor[batch][oc][out_h][out_w] = Y_tensor[h][w][oc][tile];
        }
      }
    }
  }
}

void fused_sgemm(const int64_t tile_in_h,
                 const int64_t tile_in_w,
                 const int64_t num_tiles,
                 const int64_t oc,
                 const int64_t ic,
                 float *U,  
                 float *V,
                 float *M) {
  typedef float (*U_tensor_t)[tile_in_w][oc][ic];
  typedef float (*V_tensor_t)[tile_in_w][num_tiles][ic];
  typedef float (*M_tensor_t)[tile_in_w][oc][num_tiles];

  U_tensor_t U_tensor = (U_tensor_t)U;
  V_tensor_t V_tensor = (V_tensor_t)V;
  M_tensor_t M_tensor = (M_tensor_t)M;

  #pragma omp parallel for collapse(4) schedule(static)
  for (int64_t h = 0; h < tile_in_h; h++) {
    for (int64_t w = 0; w < tile_in_w; w++) {
      for (int64_t m = 0; m < num_tiles; m++) {
        for (int64_t n = 0; n < oc; n++) {
          float sum = 0.0f;
          #pragma omp simd reduction(+:sum)
          for (int64_t k = 0; k < ic; k++) {
            sum += V_tensor[h][w][m][k] * U_tensor[h][w][n][k];
          }
          M_tensor[h][w][n][m] = sum;
        }
      }
    }
  }
}


// 4M
#define SGEMM_SMALL_THRESHOLD 4 * 1024 * 1024

void sgemm(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C) {
  typedef float(*A_tensor_t)[K];
  typedef float(*B_tensor_t)[K];
  typedef float(*C_tensor_t)[M];
  A_tensor_t A_tensor = (A_tensor_t)A;
  B_tensor_t B_tensor = (B_tensor_t)B;
  C_tensor_t C_tensor = (C_tensor_t)C;

  if (M * N * K < SGEMM_SMALL_THRESHOLD) {
    // small matrix, no omp
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int64_t k = 0; k < K; ++k) {
          // TIPS: localize the memory access
          sum += A_tensor[m][k] * B_tensor[n][k];
        }
        C_tensor[n][m] = sum;
      }
    }
  }else{
    // large matrix, use omp
    // printf("large matrix: M: %ld, N: %ld, K: %ld, M * N * K: %ld\n", M, N, K, M * N * K);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int64_t k = 0; k < K; ++k) {
          sum += A_tensor[m][k] * B_tensor[n][k];
        }
        C_tensor[n][m] = sum;
      }
    }
  }
}

void winograd_convolution(
    float *__restrict__ image, /**< float [batch_num][input_channel_num][image_height][image_width] */
    const int image_height,
    const int image_width,
    const int input_channel_num,
    float *__restrict__ filter, /**< float [output_channel_num][input_channel_num][FLT_H][FLT_W] */
    const int output_channel_num,
    const int batch_num,
    float *__restrict__ out) {
  /* new vars of shape */
  const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
  const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
  const out_shape_t os = get_output_shape(is, fs);
  const tiling_info_t ti = get_tiling_info(is, os);
  const U_shape_t us = get_U_shape(fs, ti);
  const V_shape_t vs = get_V_shape(is, ti);

  float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
  float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
  float *U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
  float *V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
  float *M = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
  float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

  #ifdef DEBUG
    int64_t start_time = current_time_ms();
  #endif

  filter_packing(filter, packed_filter, fs);
  #ifdef DEBUG
    int64_t filter_packing_time = current_time_ms();
  #endif

  filter_transform(packed_filter, U, fs, us, us.oc * us.ic);
  #ifdef DEBUG
    int64_t filter_transform_time = current_time_ms();
  #endif

  image_packing(image, packed_image, is, ti);
  #ifdef DEBUG
    int64_t image_packing_time = current_time_ms();
  #endif

  image_transform(packed_image, V, vs, ti, vs.ic * vs.num_tiles);
  #ifdef DEBUG
    int64_t image_transform_time = current_time_ms();  
  #endif

  // fused sgemm
  fused_sgemm(ti.tile_in_h, ti.tile_in_w, ti.num_tiles, us.oc, us.ic, U, V, M);

  // typedef float(*U_tensor_t)[ti.tile_in_w][us.oc][us.ic];
  // typedef float(*V_tensor_t)[ti.tile_in_w][vs.num_tiles][vs.ic];
  // typedef float(*M_tensor_t)[ti.tile_in_w][us.oc][vs.num_tiles];
  // U_tensor_t U_tensor = (U_tensor_t)U;
  // V_tensor_t V_tensor = (V_tensor_t)V;
  // M_tensor_t M_tensor = (M_tensor_t)M;
  // int64_t local_num_tiles = vs.num_tiles;
  // int64_t local_oc = us.oc;
  // int64_t local_ic = us.ic;
  // #pragma omp parallel for collapse(2) schedule(static)
  // for (int64_t h = 0; h < ti.tile_in_h; ++h) {
  //   for (int64_t w = 0; w < ti.tile_in_w; ++w) {
  //     sgemm(local_num_tiles,
  //           local_oc,
  //           local_ic,
  //           (float *)(V_tensor[h][w]),
  //           (float *)(U_tensor[h][w]),
  //           (float *)(M_tensor[h][w]));
  //   }
  // }
  
  #ifdef DEBUG
    int64_t sgemm_time = current_time_ms();
  #endif

  output_transform(M, Y, ti, us.oc * vs.num_tiles);
  #ifdef DEBUG
    int64_t output_transform_time = current_time_ms();
  #endif

  output_unpacking_store(Y, out, os, ti);
  #ifdef DEBUG
    int64_t output_unpacking_store_time = current_time_ms();
  #endif

  #ifdef DEBUG
    printf("--------------------------------\n");
    printf("filter_packing_time: %ld, delta: %ldms\n", filter_packing_time, filter_packing_time - start_time);
    printf("filter_transform_time: %ld, delta: %ldms\n", filter_transform_time, filter_transform_time - filter_packing_time);
    printf("image_packing_time: %ld, delta: %ldms\n", image_packing_time, image_packing_time - filter_transform_time);
    printf("image_transform_time: %ld, delta: %ldms\n", image_transform_time, image_transform_time - image_packing_time);
    printf("num of sgemm times: %ld, h: %ld, w: %ld\n", ti.tile_in_h * ti.tile_in_w, ti.tile_in_h, ti.tile_in_w);
    printf("M(num_tiles) x N(oc) x K(ic): %ld x %ld x %ld\n", ti.num_tiles, us.oc, us.ic);
    printf("sgemm_time: %ld, delta: %ldms\n", sgemm_time, sgemm_time - image_transform_time);
    printf("output_transform_time: %ld, delta: %ldms\n", output_transform_time, output_transform_time - sgemm_time);
    printf("output_unpacking_store_time: %ld, delta: %ldms\n", output_unpacking_store_time, output_unpacking_store_time - output_transform_time);
    printf("--------------------------------\n");
  #endif

  free(packed_filter);
  free(packed_image);
  free(U);
  free(V);
  free(M);
  free(Y);
}