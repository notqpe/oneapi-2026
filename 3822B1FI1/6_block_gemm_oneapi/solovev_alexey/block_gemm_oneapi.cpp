#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    size_t size,
    sycl::device device) {

    const size_t N = size;
    const size_t TILE = 16;

    sycl::queue q(device);

    float* A = sycl::malloc_shared<float>(N * N, q);
    float* B = sycl::malloc_shared<float>(N * N, q);
    float* C = sycl::malloc_shared<float>(N * N, q);

    for (size_t i = 0; i < N * N; i++) {
        A[i] = a[i];
        B[i] = b[i];
        C[i] = 0.0f;
    }

    sycl::range<2> global(N, N);
    sycl::range<2> local(TILE, TILE);

    q.submit([&](sycl::handler& h) {

        sycl::local_accessor<float, 2> tileA({ TILE, TILE }, h);
        sycl::local_accessor<float, 2> tileB({ TILE, TILE }, h);

        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) {

                size_t row = item.get_global_id(0);
                size_t col = item.get_global_id(1);

                size_t local_row = item.get_local_id(0);
                size_t local_col = item.get_local_id(1);

                float sum = 0.0f;

                for (size_t t = 0; t < N; t += TILE) {

                    tileA[local_row][local_col] = A[row * N + t + local_col];
                    tileB[local_row][local_col] = B[(t + local_row) * N + col];

                    item.barrier(sycl::access::fence_space::local_space);

                    for (size_t k = 0; k < TILE; k++)
                        sum += tileA[local_row][k] * tileB[k][local_col];

                    item.barrier(sycl::access::fence_space::local_space);
                }

                C[row * N + col] = sum;
            });
        }).wait();

    std::vector<float> result(N * N);

    for (size_t i = 0; i < N * N; i++)
        result[i] = C[i];

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);

    return result;
}