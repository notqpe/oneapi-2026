#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    using ExecSpace = Kokkos::SYCL;

    const float h = (end - start) / static_cast<float>(count);

    float integral_sin = 0.0f;
    float integral_cos = 0.0f;

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecSpace>(0, count),
        KOKKOS_LAMBDA(int i, float& local_sum) {
            float x = start + (i + 0.5f) * h;
            local_sum += sinf(x);
        },
        integral_sin
    );

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecSpace>(0, count),
        KOKKOS_LAMBDA(int j, float& local_sum) {
            float y = start + (j + 0.5f) * h;
            local_sum += cosf(y);
        },
        integral_cos
    );

    Kokkos::fence();

    return integral_sin * integral_cos * h * h;
}