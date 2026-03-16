#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    if (count <= 0) {
        return 0.0f;
    }

    using ExecutionSpace = Kokkos::SYCL;
    using MemorySpace = Kokkos::SYCLDeviceUSMSpace;

    const float dx = (end - start) / static_cast<float>(count);
    const float dy = dx;

    float result = 0.0f;

    Kokkos::parallel_reduce(
        "DoubleIntegralMidpoint",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
            {{0, 0}},
            {{count, count}}
        ),
        KOKKOS_LAMBDA(const int iy, const int ix, float& lsum) {
            const float x_mid = start + (static_cast<float>(ix) + 0.5f) * dx;
            const float y_mid = start + (static_cast<float>(iy) + 0.5f) * dy;

            const float f = Kokkos::sin(x_mid) * Kokkos::cos(y_mid);

            lsum += f * dx * dy;
        },
        result
    );

    return result;
}