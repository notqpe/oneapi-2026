#include "acc_jacobi_oneapi.h"

std::vector<float> JacobiAccONEAPI(const std::vector<float>& a,
                                   const std::vector<float>& b, float accuracy,
                                   sycl::device device) {
  sycl::queue queue(device);

  const size_t n = b.size();

  std::vector<float> x_old(n, 0.0f);
  std::vector<float> x_new(n, 0.0f);

  std::vector<float> inv_diag(n);
  for (size_t i = 0; i < n; ++i) {
    inv_diag[i] = 1.0f / a[i * n + i];
  }

  sycl::buffer<float> a_buf(a.data(), n * n);
  sycl::buffer<float> b_buf(b.data(), n);
  sycl::buffer<float> x_old_buf(x_old.data(), n);
  sycl::buffer<float> x_new_buf(x_new.data(), n);
  sycl::buffer<float> inv_buf(inv_diag.data(), n);

  float error = 0.0f;
  sycl::buffer<float> err_buf(&error, 1);

  const size_t local_size = 128;
  const size_t global_size = ((n + local_size - 1) / local_size) * local_size;

  for (int iter = 0; iter < ITERATIONS; ++iter) {
    error = 0.0f;

    queue.submit([&](sycl::handler& h) {
      auto A = a_buf.get_access<sycl::access::mode::read>(h);
      auto B = b_buf.get_access<sycl::access::mode::read>(h);
      auto X_old = x_old_buf.get_access<sycl::access::mode::read>(h);
      auto X_new = x_new_buf.get_access<sycl::access::mode::write>(h);
      auto INV = inv_buf.get_access<sycl::access::mode::read>(h);

      auto err_red = sycl::reduction(err_buf, h, sycl::plus<float>());

      sycl::local_accessor<float, 1> x_tile(local_size, h);

      h.parallel_for(sycl::nd_range<1>(global_size, local_size), err_red,
                     [=](sycl::nd_item<1> item, auto& err_sum) {
                       size_t i = item.get_global_id(0);
                       size_t lid = item.get_local_id(0);

                       if (i >= n) return;

                       float sigma = 0.0f;

                       for (size_t tile = 0; tile < n; tile += local_size) {
                         size_t j = tile + lid;

                         x_tile[lid] = (j < n) ? X_old[j] : 0.0f;

                         item.barrier(sycl::access::fence_space::local_space);

                         for (size_t k = 0; k < local_size; ++k) {
                           size_t col = tile + k;

                           if (col < n) {
                             sigma += A[i * n + col] * x_tile[k];
                           }
                         }

                         item.barrier(sycl::access::fence_space::local_space);
                       }

                       sigma -= A[i * n + i] * X_old[i];

                       float new_val = (B[i] - sigma) * INV[i];
                       X_new[i] = new_val;

                       float diff = sycl::fabs(new_val - X_old[i]);
                       err_sum += diff;
                     });
    });

    queue.wait();

    if (error < accuracy) break;

    std::swap(x_old_buf, x_new_buf);
  }

  {
    auto res = x_old_buf.get_host_access();
    for (size_t i = 0; i < n; ++i) {
      x_old[i] = res[i];
    }
  }

  return x_old;
}