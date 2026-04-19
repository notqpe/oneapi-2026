#include "mkl_gemm_oneapi.h"

#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(const std::vector<float>& a,
                                 const std::vector<float>& b, size_t size,
                                 sycl::device device) {
  sycl::queue compute_queue(
      device, sycl::property_list{sycl::property::queue::in_order{},
                                  sycl::property::queue::enable_profiling{}});

  const int n = static_cast<int>(size);
  const size_t elements = static_cast<size_t>(n) * n;

  float* device_a = sycl::malloc_device<float>(elements, compute_queue);
  float* device_b = sycl::malloc_device<float>(elements, compute_queue);
  float* device_c = sycl::malloc_device<float>(elements, compute_queue);

  sycl::event copy_a =
      compute_queue.memcpy(device_a, a.data(), sizeof(float) * elements);

  sycl::event copy_b =
      compute_queue.memcpy(device_b, b.data(), sizeof(float) * elements);

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  auto nontrans = oneapi::mkl::transpose::nontrans;

  sycl::event gemm_event = oneapi::mkl::blas::gemm(
      compute_queue, nontrans, nontrans, n, n, n, alpha, device_b, n, device_a,
      n, beta, device_c, n, {copy_a, copy_b});

  std::vector<float> result(elements);

  sycl::event copy_back = compute_queue.memcpy(
      result.data(), device_c, sizeof(float) * elements, {gemm_event});

  copy_back.wait();

  sycl::free(device_a, compute_queue);
  sycl::free(device_b, compute_queue);
  sycl::free(device_c, compute_queue);

  return result;
}