#include "dev_jacobi_oneapi.h"

std::vector<float> JacobiDevONEAPI(const std::vector<float>& a,
                                   const std::vector<float>& b, float accuracy,
                                   sycl::device device) {
  sycl::queue queue(device);

  const size_t n = b.size();

  float* A = sycl::malloc_device<float>(n * n, queue);
  float* B = sycl::malloc_device<float>(n, queue);
  float* x_old = sycl::malloc_device<float>(n, queue);
  float* x_new = sycl::malloc_device<float>(n, queue);
  float* inv_diag = sycl::malloc_device<float>(n, queue);
  float* error = sycl::malloc_device<float>(1, queue);

  queue.memcpy(A, a.data(), sizeof(float) * n * n);
  queue.memcpy(B, b.data(), sizeof(float) * n);
  queue.memset(x_old, 0, sizeof(float) * n);

  queue
      .submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
          inv_diag[i] = 1.0f / A[i * n + i];
        });
      })
      .wait();

  const int local_size = 256;
  const int global_size = ((n + local_size - 1) / local_size) * local_size;

  for (int iter = 0; iter < ITERATIONS; ++iter) {
    float zero = 0.0f;
    queue.memcpy(error, &zero, sizeof(float));

    queue.submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> local_err(local_size, h);

      h.parallel_for(
          sycl::nd_range<1>(global_size, local_size),
          [=](sycl::nd_item<1> item) {
            const int i = item.get_global_id(0);
            const int lid = item.get_local_id(0);

            float diff = 0.0f;

            if (i < n) {
              const float* row = A + i * n;
              const float xi = x_old[i];

              float sigma = 0.0f;

              size_t j = 0;
              for (; j + 4 <= n; j += 4) {
                sigma += row[j] * x_old[j];
                sigma += row[j + 1] * x_old[j + 1];
                sigma += row[j + 2] * x_old[j + 2];
                sigma += row[j + 3] * x_old[j + 3];
              }
              for (; j < n; ++j) {
                sigma += row[j] * x_old[j];
              }

              sigma -= row[i] * xi;

              const float new_val = (B[i] - sigma) * inv_diag[i];

              diff = sycl::fabs(new_val - xi);

              x_new[i] = new_val;
            }

            local_err[lid] = diff;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = local_size / 2; stride > 0; stride /= 2) {
              if (lid < stride) {
                local_err[lid] =
                    sycl::fmax(local_err[lid], local_err[lid + stride]);
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) {
              sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  atomic_err(*error);

              atomic_err.fetch_max(local_err[0]);
            }
          });
    });

    queue.wait();

    float host_error;
    queue.memcpy(&host_error, error, sizeof(float)).wait();

    std::swap(x_old, x_new);

    if (host_error < accuracy) break;
  }

  std::vector<float> result(n);
  queue.memcpy(result.data(), x_old, sizeof(float) * n).wait();

  sycl::free(A, queue);
  sycl::free(B, queue);
  sycl::free(x_old, queue);
  sycl::free(x_new, queue);
  sycl::free(inv_diag, queue);
  sycl::free(error, queue);

  return result;
}