#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {

    const int n = b.size();

    Kokkos::View<float*> x("x", n);
    Kokkos::View<float*> x_new("x_new", n);

    Kokkos::View<float*> vec_b("b", n);
    Kokkos::View<float*> mat_a("a", n * n);

    auto host_b = Kokkos::create_mirror_view(vec_b);
    auto host_a = Kokkos::create_mirror_view(mat_a);

    for (int i = 0; i < n; ++i)
        host_b(i) = b[i];

    for (int i = 0; i < n * n; ++i)
        host_a(i) = a[i];

    Kokkos::deep_copy(vec_b, host_b);
    Kokkos::deep_copy(mat_a, host_a);

    Kokkos::deep_copy(x, 0.0f);

    for (int iter = 0; iter < ITERATIONS; ++iter) {

        Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(const int i) {

                float sigma = 0.0f;
                int row = i * n;

                for (int j = 0; j < n; ++j) {
                    if (j != i)
                        sigma += mat_a(row + j) * x(j);
                }

                x_new(i) = (vec_b(i) - sigma) / mat_a(row + i);
            }
        );

        float max_diff = 0.0f;

        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(const int i, float& local_max) {
                float diff = Kokkos::fabs(x_new(i) - x(i));
                if (diff > local_max) local_max = diff;
            },
            Kokkos::Max<float>(max_diff)
        );

        if (max_diff < accuracy)
            break;

        Kokkos::kokkos_swap(x, x_new);
    }

    auto host_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(host_x, x);

    std::vector<float> result(n);

    for (int i = 0; i < n; ++i)
        result[i] = host_x(i);

    return result;
}
