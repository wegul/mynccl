#pragma once

#include "config.h"
#include <cstddef>
#include <cstdint>
#include <infiniband/verbs.h>
#include <sys/mman.h>

class BufPool {
protected:
  void *base_addr_;

  uint32_t nr_elements_;
  size_t element_size_;
  struct ibv_mr *mr_;
  uint64_t *buffer_pool_;

public:
  uint32_t head_;
  uint32_t tail_;


  BufPool(uint32_t nr_elements, size_t element_size,
          struct ibv_mr *mr = nullptr)
      : nr_elements_(nr_elements), element_size_(element_size), mr_(mr) {
    if (mr_) {
      base_addr_ = mr_->addr;
      buffer_pool_ = (uint64_t *)mr_->addr;
    } else {
      base_addr_ =
          mmap(nullptr, nr_elements * element_size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (base_addr_ == MAP_FAILED) {
        ERROR("Failed to allocate memory for BufPool");
      }
      buffer_pool_ = new uint64_t[nr_elements];
    }
    head_ = tail_ = 0;
    // Reserve one element for distinguished empty/full state.
    for (uint32_t i = 0; i < nr_elements_ - 1; i++) {
      free_buff((uint64_t)base_addr_ + i * element_size_);
    }
  }
  ~BufPool() {
    if (!mr_) {
      munmap(base_addr_, nr_elements_ * element_size_);
    }
    delete[] buffer_pool_;
  }

  inline bool full(void) { return ((tail_ + 1) & (nr_elements_ - 1)) == head_; }

  inline bool empty(void) { return head_ == tail_; }

  inline uint32_t size(void) { return (tail_ - head_) & (nr_elements_ - 1); }

  inline uint32_t get_lkey(void) {
    if (!mr_)
      return 0;
    return mr_->lkey;
  }
  inline int alloc_buff(uint64_t *buff_addr) {
    if (empty()) {
      return -1;
    }
    *buff_addr = (uint64_t)base_addr_ + buffer_pool_[head_];
    head_ = (head_ + 1) & (nr_elements_ - 1);
    return 0;
  }

  inline void free_buff(uint64_t buff_addr) {
    if (full())
      return;
    buff_addr -= (uint64_t)base_addr_;
    buffer_pool_[tail_] = buff_addr;
    tail_ = (tail_ + 1) & (nr_elements_ - 1);
  }
};