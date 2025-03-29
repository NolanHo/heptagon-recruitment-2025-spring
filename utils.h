#pragma once
#include <assert.h>
#include <stdint.h>

#define MIN(A, B) (((A) < (B)) ? (A) : (B))
#define MAX(A, B) (((A) > (B)) ? (A) : (B))

// 卷积核大小 = 3 x 3
#define FLT_H 3L
#define FLT_W 3L

// 输入tile大小 = 6 x 6
#define TILE_IN_H 6L
#define TILE_IN_W 6L

// 输出tile大小 = 4 x 4
#define TILE_OUT_H 4L
#define TILE_OUT_W 4L

#define ROUND(A, B) ((A) / (B) * (B))
#define ROUND_UP(A, B) (((A) + (B)-1) / (B) * (B))

#define DIVIDE(A, B) ((A) / (B))
#define DIVIDE_UP(A, B) (((A) + (B)-1) / (B))
#define DIV(A, B) ((A) / (B))
#define DIV_UP(A, B) (((A) + (B)-1) / (B))

/**
 * @brief Structure representing tile indices.
 */
typedef struct {
  int64_t b;  /**< Batch index. */
  int64_t th; /**< Tile index in height. */
  int64_t tw; /**< Tile index in width. */
} tile_index_t;

/**
 * @brief Structure representing the shape of a filter.
 */
typedef struct {
  int64_t oc; /**< Number of output channels. */
  int64_t ic; /**< Number of input channels. */
  int64_t h;  /**< Height of the filter. */
  int64_t w;  /**< Width of the filter. */
} filter_shape_t;

/**
 * @brief Represents the shape of an image.
 */
typedef struct {
  int64_t bs; /**< The number of images in the batch (batch size). */
  int64_t ic; /**< The number of input channels. */
  int64_t h;  /**< The height of the image. */
  int64_t w;  /**< The width of the image. */
} image_shape_t;

/**
 * @brief Represents the shape of a U matrix.
 */
typedef struct {
  int64_t h;  /**< The height of the tensor. */
  int64_t w;  /**< The width of the tensor. */
  int64_t oc; /**< The number of output channels. */
  int64_t ic; /**< The number of input channels. */
} U_shape_t;

/**
 * @brief Structure representing the shape of a V tensor.
 */
typedef struct {
  int64_t h;         /**< The height of the tensor. */
  int64_t w;         /**< The width of the tensor. */
  int64_t num_tiles; /**< The number of tiles total. */
  int64_t ic;        /**< The number of input channels. */
} V_shape_t;

/**
 * @brief Represents the output shape of a computation.
 */
typedef struct {
  int64_t bs; /**< The batch size. */
  int64_t oc; /**< The number of output channels. */
  int64_t h;  /**< The height of the output image. */
  int64_t w;  /**< The width of the output image. */
} out_shape_t;

/**
 * @brief Structure representing tiling information.
 */
typedef struct {
  int64_t bs;                 /**< The batch size. */
  int64_t num_tile_per_image; /**< The number of tiles per image. */
  int64_t num_tiles;          /**< The total number of tiles. */
  int64_t tiles_on_h;         /**< The number of tiles on the height dimension. */
  int64_t tiles_on_w;         /**< The number of tiles on the width dimension. */
  int64_t tile_in_h;          /**< The height of the input tile. */
  int64_t tile_in_w;          /**< The width of the input tile. */
  int64_t tile_out_h;         /**< The height of the output tile. */
  int64_t tile_out_w;         /**< The width of the output tile. */
} tiling_info_t;

inline out_shape_t get_output_shape(image_shape_t is, filter_shape_t fs) {
  out_shape_t os;
  os.bs = is.bs;
  os.oc = fs.oc;
  os.h = is.h - fs.h + 1;
  os.w = is.w - fs.w + 1;
  return os;
}

inline tiling_info_t get_tiling_info(image_shape_t is, out_shape_t os) {
  tiling_info_t ts;
  ts.tiles_on_h = DIV_UP(os.h, TILE_OUT_H);
  ts.tiles_on_w = DIV_UP(os.w, TILE_OUT_W);
  ts.bs = is.bs;
  ts.num_tile_per_image = ts.tiles_on_h * ts.tiles_on_w;
  ts.num_tiles = ts.num_tile_per_image * ts.bs;
  ts.tile_in_h = TILE_IN_H;
  ts.tile_in_w = TILE_IN_W;
  ts.tile_out_h = TILE_OUT_H;
  ts.tile_out_w = TILE_OUT_W;
  return ts;
}

inline U_shape_t get_U_shape(filter_shape_t fs, tiling_info_t ti) {
  U_shape_t us;
  us.oc = fs.oc;
  us.ic = fs.ic;
  us.h = ti.tile_in_h;
  us.w = ti.tile_in_w;
  return us;
}

inline V_shape_t get_V_shape(image_shape_t is, tiling_info_t ti) {
  V_shape_t vs;
  vs.num_tiles = ti.num_tiles;
  vs.ic = is.ic;
  vs.h = ti.tile_in_h;
  vs.w = ti.tile_in_w;
  return vs;
}

inline tile_index_t get_tile_index(int64_t tile, tiling_info_t ts) {
  tile_index_t ti;
  ti.b = tile / ts.num_tile_per_image;
  tile = tile % ts.num_tile_per_image;
  ti.th = tile / ts.tiles_on_w;
  ti.tw = tile % ts.tiles_on_w;
  return ti;
}
