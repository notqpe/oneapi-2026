#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    const float dx = (end - start) / count;
    const float dy = (end - start) / count;
    const float area = dx * dy;

    float result = 0.0f;

    Kokkos::parallel_reduce(
        "IntegralSum",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(const int i, const int j, float& local_sum) {
            float x = start + (i + 0.5f) * dx;
            float y = start + (j + 0.5f) * dy;

            float val = sinf(x) * cosf(y);

            local_sum += val;
        },
        result);

    return result * area;
}