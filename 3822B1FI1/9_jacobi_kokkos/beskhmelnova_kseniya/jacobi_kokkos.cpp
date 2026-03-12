#include "jacobi_kokkos.h"
#include <Kokkos_Core.hpp>
#include <cmath>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    
    using ExecSpace = Kokkos::SYCL;
    using MemSpace = Kokkos::SYCLDeviceUSMSpace;

    const int n = static_cast<int>(b.size());
    const float accuracy_sq = accuracy * accuracy;

    Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace> d_a("A", n, n);
    Kokkos::View<float*, MemSpace> d_b("b", n);
    Kokkos::View<float*, MemSpace> d_inv("inv_diag", n);
    Kokkos::View<float*, MemSpace> x_curr("x_curr", n);
    Kokkos::View<float*, MemSpace> x_next("x_next", n);

    auto h_a = Kokkos::create_mirror_view(d_a);
    auto h_b = Kokkos::create_mirror_view(d_b);
    for (int i = 0; i < n; i++) {
        h_b(i) = b[i];
        for (int j = 0; j < n; j++) {
            h_a(i, j) = a[i * n + j];
        }
    }
    Kokkos::deep_copy(d_a, h_a);
    Kokkos::deep_copy(d_b, h_b);

    Kokkos::parallel_for("InitDiag", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(int i) {
            d_inv(i) = 1.0f / d_a(i, i);
            x_curr(i) = 0.0f;
        });

    bool converged = false;
    const int CHECK_INTERVAL = 8;

    for (int iter = 0; iter < ITERATIONS && !converged; iter++) {
        Kokkos::parallel_for("JacobiUpdate", Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(int i) {
                float sum = 0.0f;
                for (int j = 0; j < n; j++) {
                    if (j != i) {
                        sum += d_a(i, j) * x_curr(j);
                    }
                }
                x_next(i) = d_inv(i) * (d_b(i) - sum);
            });

        if ((iter + 1) % CHECK_INTERVAL == 0) {
            float max_diff = 0.0f;
            Kokkos::parallel_reduce("NormCheck",
                Kokkos::RangePolicy<ExecSpace>(0, n),
                KOKKOS_LAMBDA(int i, float& local_max) {
                    float diff = Kokkos::fabs(x_next(i) - x_curr(i));
                    if (diff > local_max) local_max = diff;
                },
                Kokkos::Max<float>(max_diff)
            );

            if (max_diff < accuracy) {
                converged = true;
                break;
            }
        }

        Kokkos::kokkos_swap(x_curr, x_next);
    }

    std::vector<float> result(n);
    auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_curr);
    for (int i = 0; i < n; i++) {
        result[i] = h_x(i);
    }

    return result;
}
