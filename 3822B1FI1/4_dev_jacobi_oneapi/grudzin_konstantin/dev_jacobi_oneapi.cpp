#include "dev_jacobi_oneapi.h"
#include <algorithm>

using buftype = sycl::buffer<float>;

std::vector<float> JacobiDevONEAPI(const std::vector<float> &a,
                                   const std::vector<float> &b, float accuracy,
                                   sycl::device device) {
  size_t size = b.size();
  std::vector<float> previous(size, 0.0f);
  std::vector<float> result(size, 0.0f);

  auto stop = [&]() -> bool {
    float norm = 0.0f;
    for (int index = 0; index < previous.size(); ++index) {
      float diff = result[index] - previous[index];
      norm += diff * diff;
    }
    return norm < accuracy * accuracy;
  };

  sycl::queue queue(device);

  auto alloc = [&queue](size_t size) -> float * {
    return sycl::malloc_device<float>(size, queue);
  };

  auto free = [&queue](void *ptr) -> void { sycl::free(ptr, queue); };

  float *in_a = alloc(a.size());
  float *in_b = alloc(b.size());
  float *in_previous = alloc(previous.size());
  float *in_result = alloc(result.size());

  queue.memcpy(in_a, a.data(), sizeof(float) * a.size()).wait();
  queue.memcpy(in_b, b.data(), sizeof(float) * b.size()).wait();

  for (int epoch = 0; epoch < ITERATIONS; ++epoch) {

    queue.memcpy(in_previous, previous.data(), sizeof(float) * size).wait();

    queue
        .submit([&](sycl::handler &handler) {
          handler.parallel_for(sycl::range<1>(size), [=](sycl::id<1> id) {
            int index = id.get(0);
            float next_result = 0;

            for (int idx = 0; idx < size; idx++) {
              if (index == idx) {
                next_result += in_b[idx];
              } else {
                next_result -= in_a[index * size + idx] * in_previous[idx];
              }
            }
            next_result /= in_a[index * size + index];
            in_result[index] = next_result;
          });
        })
        .wait();

    queue.memcpy(result.data(), in_result, sizeof(float) * size).wait();

    if (stop()) {
      break;
    }

    std::swap(previous, result);
  }

  free(in_a);
  free(in_b);
  free(in_previous);
  free(in_result);

  return result;
}
