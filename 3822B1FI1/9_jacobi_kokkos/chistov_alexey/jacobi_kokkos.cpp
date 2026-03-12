#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(const std::vector<float>& a,
                                const std::vector<float>& b,
                                float accuracy) {
    using ExecSpace = Kokkos::SYCL;
    using MemSpace  = Kokkos::SYCLDeviceUSMSpace;

    const int n = b.size();

    Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace> d_a("A", n, n);
    Kokkos::View<float*, MemSpace> d_b("b", n);
    Kokkos::View<float*, MemSpace> x_old("x_old", n);
    Kokkos::View<float*, MemSpace> x_new("x_new", n);
    Kokkos::View<float*, MemSpace> inv_diag("inv_diag", n);

    auto h_a = Kokkos::create_mirror_view(d_a);
    auto h_b = Kokkos::create_mirror_view(d_b);

    for (int i = 0; i < n; ++i) {
        h_b(i) = b[i];
        for (int j = 0; j < n; ++j)
            h_a(i, j) = a[i * n + j];
    }

    Kokkos::deep_copy(d_a, h_a);
    Kokkos::deep_copy(d_b, h_b);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(int i) {
            inv_diag(i) = 1.0f / d_a(i, i);
            x_old(i) = 0.0f;
        });

    bool converged = false;
    const int CHECK_INTERVAL = 8;

    for (int iter = 0; iter < ITERATIONS && !converged; ++iter) {

        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(int i) {

                float sigma = 0.0f;

                for (int j = 0; j < n; ++j) {
                    if (j != i)
                        sigma += d_a(i, j) * x_old(j);
                }

                x_new(i) = (d_b(i) - sigma) * inv_diag(i);
            });

        if ((iter + 1) % CHECK_INTERVAL == 0) {

            float error = 0.0f;

            Kokkos::parallel_reduce(
                Kokkos::RangePolicy<ExecSpace>(0, n),
                KOKKOS_LAMBDA(int i, float& local_max) {

                    float diff = Kokkos::fabs(x_new(i) - x_old(i));
                    if (diff > local_max)
                        local_max = diff;
                },
                Kokkos::Max<float>(error)
            );

            if (error < accuracy) {
                converged = true;
                break;
            }
        }

        Kokkos::kokkos_swap(x_old, x_new);
    }

    auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_old);

    std::vector<float> result(n);
    for (int i = 0; i < n; ++i)
        result[i] = h_x(i);

    return result;
}