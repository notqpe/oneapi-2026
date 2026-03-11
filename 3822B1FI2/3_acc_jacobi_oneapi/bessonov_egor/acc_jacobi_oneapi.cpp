#include "acc_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiAccONEAPI(
  const std::vector<float>& a,
  const std::vector<float>& b,
  float accuracy,
  sycl::device device) {

  const int n = b.size();

  std::vector<float> prev(n, 0.0f);
  std::vector<float> curr(n, 0.0f);

  sycl::buffer<float> a_buf(a.data(), a.size());
  sycl::buffer<float> b_buf(b.data(), b.size());
  sycl::buffer<float> prev_buf(prev.data(), prev.size());
  sycl::buffer<float> curr_buf(curr.data(), curr.size());

  sycl::queue q(device);

  for (int iter = 0; iter < ITERATIONS; ++iter) {
    q.submit([&](sycl::handler& cgh) {
      auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
      auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
      auto prev_acc = prev_buf.get_access<sycl::access::mode::read>(cgh);
      auto curr_acc = curr_buf.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        int i = id[0];
        float value = b_acc[i];

        for (int j = 0; j < n; ++j) {
          if (i != j) {
            value -= a_acc[i * n + j] * prev_acc[j];
          }
        }

        curr_acc[i] = value / a_acc[i * n + i];
        });
      }).wait();

    bool ok = true;
    {
      auto prev_acc = prev_buf.get_host_access();
      auto curr_acc = curr_buf.get_host_access();

      for (int i = 0; i < n; ++i) {
        if (std::fabs(curr_acc[i] - prev_acc[i]) >= accuracy) {
          ok = false;
        }
        prev_acc[i] = curr_acc[i];
      }
    }

    if (ok) {
      break;
    }
  }

  return prev;
}