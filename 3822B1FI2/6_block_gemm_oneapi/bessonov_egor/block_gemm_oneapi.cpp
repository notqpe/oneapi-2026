#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
  const std::vector<float>& a,
  const std::vector<float>& b,
  size_t size,
  sycl::device device) {

  constexpr size_t BLOCK_SIZE = 16;
  const size_t total = size * size;

  std::vector<float> result(total, 0.0f);

  sycl::queue q(device);

  float* a_dev = sycl::malloc_device<float>(total, q);
  float* b_dev = sycl::malloc_device<float>(total, q);
  float* c_dev = sycl::malloc_device<float>(total, q);

  q.memcpy(a_dev, a.data(), total * sizeof(float));
  q.memcpy(b_dev, b.data(), total * sizeof(float));
  q.memset(c_dev, 0, total * sizeof(float)).wait();

  sycl::range<2> global_range(size, size);
  sycl::range<2> local_range(BLOCK_SIZE, BLOCK_SIZE);

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 2> a_block(local_range, cgh);
    sycl::local_accessor<float, 2> b_block(local_range, cgh);

    cgh.parallel_for(
      sycl::nd_range<2>(global_range, local_range),
      [=](sycl::nd_item<2> item) {
        const size_t i = item.get_global_id(0);
        const size_t j = item.get_global_id(1);

        const size_t li = item.get_local_id(0);
        const size_t lj = item.get_local_id(1);

        float sum = 0.0f;

        for (size_t block = 0; block < size / BLOCK_SIZE; ++block) {
          a_block[li][lj] = a_dev[i * size + block * BLOCK_SIZE + lj];
          b_block[li][lj] = b_dev[(block * BLOCK_SIZE + li) * size + j];

          item.barrier(sycl::access::fence_space::local_space);

          for (size_t k = 0; k < BLOCK_SIZE; ++k) {
            sum += a_block[li][k] * b_block[k][lj];
          }

          item.barrier(sycl::access::fence_space::local_space);
        }

        c_dev[i * size + j] = sum;
      });
    }).wait();

  q.memcpy(result.data(), c_dev, total * sizeof(float)).wait();

  sycl::free(a_dev, q);
  sycl::free(b_dev, q);
  sycl::free(c_dev, q);

  return result;
}