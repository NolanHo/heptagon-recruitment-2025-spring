#include <cstdlib>
// #include <jemalloc/jemalloc.h>


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
    memory_manager.ptr = (float *)aligned_alloc(64, size);
    memory_manager.size = size;
    memory_manager.in_use = 1;
  }else{
    // 非第一次分配
    if (memory_manager.size < size) {
      // memory_manager.ptr = (float *)realloc(memory_manager.ptr, size);
      // free(memory_manager.ptr);
      memory_manager.ptr = (float *)aligned_alloc(64, size);
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