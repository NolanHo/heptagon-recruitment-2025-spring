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

void output_transform_locality(float *__restrict__ M,
                         float *__restrict__ Y,
                         const tiling_info_t ti,
                         const int64_t collapsed_dim_size) {
  typedef float (*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float (*Y_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  M_tensor_t M_tensor = (M_tensor_t)M;
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;

  #pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {

    float local_M[6][ti.tile_in_w];
    for (int c = 0; c < 6; c++) {
      for (int w = 0; w < ti.tile_in_w; w++) {
        local_M[c][w] = M_tensor[c][w][idx];
      }
    }

    float tempH[ti.tile_out_h][ti.tile_in_w];
    for (int w = 0; w < ti.tile_in_w; w++) {
      float z0, z1, z2, z3, z4;
      z4 = local_M[0][w];
      z0 = z4;

      z4 = local_M[1][w];
      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = local_M[2][w];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = local_M[3][w];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = local_M[4][w];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = local_M[5][w];
      z3 += z4;

      tempH[0][w] = z0;
      tempH[1][w] = z1;
      tempH[2][w] = z2;
      tempH[3][w] = z3;
    }

    float local_out[ti.tile_out_h][4];
    for (int h = 0; h < ti.tile_out_h; h++) {
      float z0, z1, z2, z3, z4;
      z4 = tempH[h][0];
      z0 = z4;

      z4 = tempH[h][1];
      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = tempH[h][2];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = tempH[h][3];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = tempH[h][4];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = tempH[h][5];
      z3 += z4;

      local_out[h][0] = z0;
      local_out[h][1] = z1;
      local_out[h][2] = z2;
      local_out[h][3] = z3;
    }

    for (int h = 0; h < ti.tile_out_h; h++) {
      Y_tensor[h][0][idx] = local_out[h][0];
      Y_tensor[h][1][idx] = local_out[h][1];
      Y_tensor[h][2][idx] = local_out[h][2];
      Y_tensor[h][3][idx] = local_out[h][3];
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

void output_unpacking_store_locality(float *__restrict__ Y,
                               float *__restrict__ out,
                               const out_shape_t os,
                               const tiling_info_t ti) {
  typedef float (*Y_tensor_t)[ti.tile_in_w][os.oc][ti.num_tiles];
  typedef float (*out_tensor_t)[os.oc][os.h][os.w];

  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  out_tensor_t out_tensor = (out_tensor_t)out;

  // pre compute tile index
  tile_index_t *tile_map = (tile_index_t *)malloc(ti.num_tiles * sizeof(tile_index_t));
  #pragma omp parallel for schedule(static)
  for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
      tile_map[tile] = get_tile_index(tile, ti);
  }

  #pragma omp parallel for collapse(3) schedule(static)
  for (int64_t h = 0; h < ti.tile_out_h; ++h) {
    for (int64_t w = 0; w < ti.tile_out_w; ++w) {
      for (int64_t oc = 0; oc < os.oc; ++oc) {
        float local_tile[ti.num_tiles];
        for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
          local_tile[tile] = Y_tensor[h][w][oc][tile];
        }

        #pragma omp simd
        for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
          tile_index_t tidx = tile_map[tile];
          int64_t batch = tidx.b;
          int64_t out_h = tidx.th * 4 + h;
          int64_t out_w = tidx.tw * 4 + w;
          if (out_h < os.h && out_w < os.w){
            out_tensor[batch][oc][out_h][out_w] = local_tile[tile];
          }
        }
      }
    }
  }

  free(tile_map);
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


void output_pipelined(float *__restrict__ M,
                      float *__restrict__ out,
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

  // pre compute tile index
  tile_index_t *tile_map = (tile_index_t *)malloc(ti.num_tiles * sizeof(tile_index_t));
  #pragma omp parallel for schedule(static)
  for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
    tile_map[tile] = get_tile_index(tile, ti);
  }

  #ifdef DEBUG_OUTPUT_PIPELINED
    int64_t get_tile_index_end = current_time_ns();
  #endif

  #pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    int oc   = idx / ti.num_tiles;
    int tile = idx % ti.num_tiles;
    

    float local_M[6][ti.tile_in_w]; // 6 * 6
    for (int c = 0; c < 6; c++) {
      for (int w = 0; w < ti.tile_in_w; w++) {
        local_M[c][w] = M_tensor[c][w][idx];
      }
    }

    float tempH[ti.tile_out_h][ti.tile_in_w];
    for (int w = 0; w < ti.tile_in_w; w++) {
      float z0, z1, z2, z3, z4;
      z4 = local_M[0][w];
      z0 = z4;

      z4 = local_M[1][w];
      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = local_M[2][w];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = local_M[3][w];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = local_M[4][w];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = local_M[5][w];
      z3 += z4;

      tempH[0][w] = z0;
      tempH[1][w] = z1;
      tempH[2][w] = z2;
      tempH[3][w] = z3;
    }

    float local_out[ti.tile_out_h][4];
    for (int h = 0; h < ti.tile_out_h; h++) {
      float z0, z1, z2, z3, z4;
      z4 = tempH[h][0];
      z0 = z4;

      z4 = tempH[h][1];
      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = tempH[h][2];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = tempH[h][3];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = tempH[h][4];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = tempH[h][5];
      z3 += z4;

      // local_out[h][0] = z0;
      // local_out[h][1] = z1;
      // local_out[h][2] = z2;
      // local_out[h][3] = z3;
      // avx 32 float * 4 = 128
      __m128 z128 = _mm_setr_ps(z0, z1, z2, z3);
      _mm_storeu_ps(&local_out[h][0], z128);
    }
    
    tile_index_t tidx = tile_map[tile];
    int batch = tidx.b;
    for (int h = 0; h < ti.tile_out_h; h++) {
      int out_h = tidx.th * ti.tile_out_w + h; 
      if (out_h >= os.h)
        continue;
      
      for (int w = 0; w < 4; w++) {
        int out_w = tidx.tw * ti.tile_out_w + w;
        if (out_w >= os.w)
          continue;
        out_tensor[batch][oc][out_h][out_w] = local_out[h][w];
      }
    }
  }
  
  free(tile_map);


  #ifdef DEBUG_OUTPUT_PIPELINED
    int64_t end_time = current_time_ns();
    printf("---time cost---\n");
    printf("get_tile_index_time: %ld ns\n", get_tile_index_end - start_time);
    printf("output_pipelined_time: %ld ns\n", end_time - get_tile_index_end);
  #endif
}

void filter_transform_pipelined(float *__restrict__ filter,
                                    float *__restrict__ V,
                                    const filter_shape_t fs,
                                    const U_shape_t us,
                                    const int64_t collapsed_dim_size) {
  typedef float (*filter_tensor_t)[fs.ic][fs.h][fs.w];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;

  typedef float (*V_tensor_t)[us.w][collapsed_dim_size];
  V_tensor_t V_tensor = (V_tensor_t)V;

  #pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    int oc = idx / fs.ic;
    int ic = idx % fs.ic;

    float g[3][3];
    for (int i = 0; i < fs.h; i++) {
      for (int j = 0; j < fs.w; j++) {
        g[i][j] = filter_tensor[oc][ic][i][j];
      }
    }

    float U_local[6][3];
    for (int j = 0; j < fs.w; j++) {
      float f0 = g[0][j];
      float f1 = g[1][j];
      float f2 = g[2][j];

      float u0 = (1.0f / 4.0f) * f0;
      float u1 = (-1.0f / 6.0f) * f0 + (-1.0f / 6.0f) * f1 + (-1.0f / 6.0f) * f2;
      float u2 = (-1.0f / 6.0f) * f0 + (1.0f / 6.0f) * f1 + (-1.0f / 6.0f) * f2;
      float u3 = (1.0f / 24.0f) * f0 + (1.0f / 12.0f) * f1 + (1.0f / 6.0f) * f2;
      float u4 = (1.0f / 24.0f) * f0 + (-1.0f / 12.0f) * f1 + (1.0f / 6.0f) * f2;
      float u5 = f2;

      U_local[0][j] = u0;
      U_local[1][j] = u1;
      U_local[2][j] = u2;
      U_local[3][j] = u3;
      U_local[4][j] = u4;
      U_local[5][j] = u5;
    }

    float V_local[6][6];
    for (int i = 0; i < 6; i++) {
      float u0 = U_local[i][0];
      float u1 = U_local[i][1];
      float u2 = U_local[i][2];

      float v0 = (1.0f / 4.0f) * u0;
      float v1 = (-1.0f / 6.0f) * u0 + (-1.0f / 6.0f) * u1 + (-1.0f / 6.0f) * u2;
      float v2 = (-1.0f / 6.0f) * u0 + (1.0f / 6.0f) * u1 + (-1.0f / 6.0f) * u2;
      float v3 = (1.0f / 24.0f) * u0 + (1.0f / 12.0f) * u1 + (1.0f / 6.0f) * u2;
      float v4 = (1.0f / 24.0f) * u0 + (-1.0f / 12.0f) * u1 + (1.0f / 6.0f) * u2;
      float v5 = u2;

      V_local[i][0] = v0;
      V_local[i][1] = v1;
      V_local[i][2] = v2;
      V_local[i][3] = v3;
      V_local[i][4] = v4;
      V_local[i][5] = v5;
    }

    for (int i = 0; i < us.h; i++) {
      for (int j = 0; j < us.w; j++) {
        V_tensor[i][j][idx] = V_local[i][j];
      }
    }
  }
}

void image_transform_pipelined(float *__restrict__ image,
                                            float *__restrict__ V,
                                            const image_shape_t is,
                                            const V_shape_t vs,
                                            const tiling_info_t ti) {
  int64_t collapsed_dim_size = vs.ic * ti.num_tiles;

  typedef float (*V_tensor_t)[6][collapsed_dim_size];
  V_tensor_t V_tensor = (V_tensor_t)V;

  typedef float (*image_tensor_t)[is.ic][is.h][is.w];
  image_tensor_t image_tensor = (image_tensor_t)image;

  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
    for (int64_t ic = 0; ic < is.ic; ic++) {
      int64_t idx = tile * is.ic + ic;
      tile_index_t tidx = get_tile_index(tile, ti);
      int batch = tidx.b;
      int base_h = tidx.th * 4;
      int base_w = tidx.tw * 4;

      float local_tile[ti.tile_in_h][ti.tile_in_w];
      for (int h = 0; h < ti.tile_in_h; h++) {
        for (int w = 0; w < ti.tile_in_w; w++) {
          int img_h = base_h + h;
          int img_w = base_w + w;
          if (img_h < is.h && img_w < is.w){
            local_tile[h][w] = image_tensor[batch][ic][img_h][img_w];
          } else {
            local_tile[h][w] = 0.0f;
          }
        }
      }

      float V_local[ti.tile_in_h][ti.tile_in_w];
      for (int w = 0; w < ti.tile_in_w; w++) {
        float z0, z1, z2, z3, z4, z5, z6;
        z6 = local_tile[0][w];
        z0 = 4.0f * z6;

        z6 = local_tile[1][w];
        z1 = -4.0f * z6;
        z2 = 4.0f * z6;
        z3 = -2.0f * z6;
        z4 = 2.0f * z6;
        z5 = 4.0f * z6;

        z6 = local_tile[2][w];
        z0 += -5.0f * z6;
        z1 += -4.0f * z6;
        z2 += -4.0f * z6;
        z3 += -z6;
        z4 += -z6;

        z6 = local_tile[3][w];
        z1 += z6;
        z2 += -z6;
        z3 += 2.0f * z6;
        z4 += -2.0f * z6;
        z5 += -5.0f * z6;

        z6 = local_tile[4][w];
        z0 += z6;
        z1 += z6;
        z2 += z6;
        z3 += z6;
        z4 += z6;

        z6 = local_tile[5][w];
        z5 += z6;

        V_local[0][w] = z0;
        V_local[1][w] = z1;
        V_local[2][w] = z2;
        V_local[3][w] = z3;
        V_local[4][w] = z4;
        V_local[5][w] = z5;
      }

      float final_tile[ti.tile_in_h][6];
      for (int h = 0; h < ti.tile_in_h; h++) {
        float z0, z1, z2, z3, z4, z5, z6;
        z6 = V_local[h][0];
        z0 = 4.0f * z6;

        z6 = V_local[h][1];
        z1 = -4.0f * z6;
        z2 = 4.0f * z6;
        z3 = -2.0f * z6;
        z4 = 2.0f * z6;
        z5 = 4.0f * z6;

        z6 = V_local[h][2];
        z0 += -5.0f * z6;
        z1 += -4.0f * z6;
        z2 += -4.0f * z6;
        z3 += -z6;
        z4 += -z6;

        z6 = V_local[h][3];
        z1 += z6;
        z2 += -z6;
        z3 += 2.0f * z6;
        z4 += -2.0f * z6;
        z5 += -5.0f * z6;

        z6 = V_local[h][4];
        z0 += z6;
        z1 += z6;
        z2 += z6;
        z3 += z6;
        z4 += z6;

        z6 = V_local[h][5];
        z5 += z6;

        final_tile[h][0] = z0;
        final_tile[h][1] = z1;
        final_tile[h][2] = z2;
        final_tile[h][3] = z3;
        final_tile[h][4] = z4;
        final_tile[h][5] = z5;
      }

      for (int h = 0; h < ti.tile_in_h; h++) {
        for (int col = 0; col < 6; col++) {
          V_tensor[h][col][idx] = final_tile[h][col];
        }
      }
    } 
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