#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  sycl::queue queue(device);

  const float step = (end - start) / static_cast<float>(count);

  float* sum_sin = sycl::malloc_shared<float>(1, queue);
  float* sum_cos = sycl::malloc_shared<float>(1, queue);

  *sum_sin = 0.0f;
  *sum_cos = 0.0f;

  const int local_size = 256;
  const int global_size = ((count + local_size - 1) / local_size) * local_size;

  queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> local_sin(local_size, h);
    sycl::local_accessor<float, 1> local_cos(local_size, h);

    h.parallel_for(
        sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
          int gid = item.get_global_id(0);
          int lid = item.get_local_id(0);

          float val_sin = 0.0f;
          float val_cos = 0.0f;

          if (gid < count) {
            float x = start + (gid + 0.5f) * step;

            val_sin = sycl::native::sin(x);
            val_cos = sycl::native::cos(x);
          }

          local_sin[lid] = val_sin;
          local_cos[lid] = val_cos;

          item.barrier(sycl::access::fence_space::local_space);

          for (int stride = local_size / 2; stride > 0; stride /= 2) {
            if (lid < stride) {
              local_sin[lid] += local_sin[lid + stride];
              local_cos[lid] += local_cos[lid + stride];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }

          if (lid == 0) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_sin(*sum_sin), atomic_cos(*sum_cos);

            atomic_sin.fetch_add(local_sin[0]);
            atomic_cos.fetch_add(local_cos[0]);
          }
        });
  });

  queue.wait();

  float result = (*sum_sin * step) * (*sum_cos * step);

  sycl::free(sum_sin, queue);
  sycl::free(sum_cos, queue);

  return result;
}