#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    const float dx = (end - start) / count;
    sycl::queue q(device);

    float sum = 0.0f;
    {
        sycl::buffer<float> sum_buf(&sum, 1);

        q.submit([&](sycl::handler& cgh) {
            auto reduction = sycl::reduction(sum_buf, cgh, sycl::plus<>());

            cgh.parallel_for(
                sycl::range<2>(count, count),
                reduction,
                [=](sycl::id<2> id, auto& sum) {
                    int i = id.get(0);
                    int j = id.get(1);
                    float x = start + dx * (i + 0.5f);
                    float y = start + dx * (j + 0.5f);
                    sum += sycl::sin(x) * sycl::cos(y);
                }
            );
        }).wait();
    }

    return sum * dx * dx;
}