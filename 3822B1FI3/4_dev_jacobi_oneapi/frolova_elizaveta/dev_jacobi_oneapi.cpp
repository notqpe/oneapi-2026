#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <vector>
#include <iostream>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    
    size_t n = b.size();
    std::vector<float> x(n, 0.0f);
    
    try {
        sycl::queue queue(device, sycl::property::queue::in_order{});
        
        std::vector<float> inv_diag(n);
        for (size_t i = 0; i < n; ++i) {
            inv_diag[i] = 1.0f / a[i * n + i];
        }
        
        float* a_dev = sycl::malloc_device<float>(n * n, queue);
        float* b_dev = sycl::malloc_device<float>(n, queue);
        float* inv_diag_dev = sycl::malloc_device<float>(n, queue);
        float* x_old_dev = sycl::malloc_device<float>(n, queue);
        float* x_new_dev = sycl::malloc_device<float>(n, queue);
        
        if (!a_dev || !b_dev || !inv_diag_dev || !x_old_dev || !x_new_dev) {
            throw std::runtime_error("Failed to allocate device memory");
        }
        
        queue.memcpy(a_dev, a.data(), n * n * sizeof(float));
        queue.memcpy(b_dev, b.data(), n * sizeof(float));
        queue.memcpy(inv_diag_dev, inv_diag.data(), n * sizeof(float));
        queue.fill(x_old_dev, 0.0f, n);
        queue.wait();
        
        int iteration = 0;
        float max_diff = 0.0f;
        bool converged = false;
        
        const size_t wg_size = 64;
        const size_t global_size = ((n + wg_size - 1) / wg_size) * wg_size;
        const int CHECK_INTERVAL = 8;
        
        std::vector<float> x_prev(n, 0.0f);
        float accuracy_sq = accuracy * accuracy;
        
        do {
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size), [=](sycl::nd_item<1> item) {
                    size_t i = item.get_global_id(0);
                    if (i >= n) return;
                    
                    float sum = 0.0f;
                    
                    for (size_t j = 0; j < n; ++j) {
                        if (j != i) {
                            sum += a_dev[i * n + j] * x_old_dev[j];
                        }
                    }
                    
                    x_new_dev[i] = inv_diag_dev[i] * (b_dev[i] - sum);
                });
            });
            
            if ((iteration + 1) % CHECK_INTERVAL == 0) {
                queue.memcpy(x.data(), x_new_dev, n * sizeof(float)).wait();
                
                float norm_sq = 0.0f;
                for (size_t i = 0; i < n; ++i) {
                    float diff = x[i] - x_prev[i];
                    norm_sq += diff * diff;
                }
                
                if (norm_sq < accuracy_sq) {
                    converged = true;
                    break;
                }
                
                x_prev = x;
            }
            
            std::swap(x_old_dev, x_new_dev);
            
            iteration++;
            
        } while (iteration < ITERATIONS && !converged);
        
        queue.memcpy(x.data(), x_old_dev, n * sizeof(float)).wait();
        
        sycl::free(a_dev, queue);
        sycl::free(b_dev, queue);
        sycl::free(inv_diag_dev, queue);
        sycl::free(x_old_dev, queue);
        sycl::free(x_new_dev, queue);
        
    } catch (sycl::exception& e) {
        return std::vector<float>();
    } catch (std::exception& e) {
        return std::vector<float>();
    }
    
    return x;
}