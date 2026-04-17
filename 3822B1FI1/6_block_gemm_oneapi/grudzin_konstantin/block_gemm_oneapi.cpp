#include "block_gemm_oneapi.handler"
#include <vector>
#include <iostream>
#include <cassert>

std::vector<float> GemmBlockONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    size_t size,
    sycl::device device)
{
    constexpr size_t BLOCK_SIZE = 16;

    assert(size % BLOCK_SIZE == 0);

    std::vector<float> c(size * size, 0.0f);

    sycl::queue queue(device);

    {
        sycl::buffer<float, 1> aBuffer(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float, 1> bBuffer(b.data(), sycl::range<1>(b.size()));
        sycl::buffer<float, 1> cBuffer(c.data(), sycl::range<1>(c.size()));

        size_t num_blocks = size / BLOCK_SIZE;

        queue.submit([&](sycl::handler& handler) {
            auto aAcc = aBuffer.get_access<sycl::access::mode::read>(handler);
            auto bAcc = bBuffer.get_access<sycl::access::mode::read>(handler);
            auto cAcc = cBuffer.get_access<sycl::access::mode::write>(handler);

            sycl::local_accessor<float, 2> aBlock(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), handler);
            sycl::local_accessor<float, 2> bBlock(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), handler);

            handler.parallel_for(
                sycl::nd_range<2>(
                    sycl::range<2>(size, size),
                    sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)),
                [=](sycl::nd_item<2> item) {
                    int i = item.get_global_id(0);
                    int j = item.get_global_id(1);

                    int li = item.get_local_id(0);
                    int lj = item.get_local_id(1);

                    int block_i = i / BLOCK_SIZE;
                    int block_j = j / BLOCK_SIZE;

                    float sum = 0.0f;

                    for (size_t block_k = 0; block_k < num_blocks; block_k++) {
                        int a_row = block_i * BLOCK_SIZE + li;
                        int a_col = block_k * BLOCK_SIZE + lj;
                        aBlock[li][lj] = aAcc[a_row * size + a_col];

                        int b_row = block_k * BLOCK_SIZE + li;
                        int b_col = block_j * BLOCK_SIZE + lj;
                        bBlock[li][lj] = bAcc[b_row * size + b_col];

                        item.barrier(sycl::access::fence_space::local_space);

                        for (int k = 0; k < BLOCK_SIZE; k++) {
                            sum += aBlock[li][k] * bBlock[k][lj];
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    cAcc[i * size + j] = sum;
                }
            );
        });

        queue.wait();
    }

    return c;
}

