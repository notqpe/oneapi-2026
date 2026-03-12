#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <vector>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    
    size_t n = b.size();
    std::vector<float> x(n, 0.0f);
    
    try {
        sycl::queue queue(device);
        
        sycl::buffer<float> a_buf(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float> b_buf(b.data(), sycl::range<1>(n));
        sycl::buffer<float> x_old_buf(n);
        sycl::buffer<float> x_new_buf(n);
        
        queue.submit([&](sycl::handler& cgh) {
            auto x_old_acc = x_old_buf.get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                x_old_acc[idx] = 0.0f;
            });
        });
        
        float max_diff = 0.0f;
        int iteration = 0;
        
        do {
            queue.submit([&](sycl::handler& cgh) {
                auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
                auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
                auto x_old_acc = x_old_buf.get_access<sycl::access::mode::read>(cgh);
                auto x_new_acc = x_new_buf.get_access<sycl::access::mode::write>(cgh);
                
                cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    size_t i = idx[0];
                    float sum = 0.0f;
                    
                    for (size_t j = 0; j < n; ++j) {
                        if (j != i) {
                            sum += a_acc[i * n + j] * x_old_acc[j];
                        }
                    }
                    
                    x_new_acc[i] = (b_acc[i] - sum) / a_acc[i * n + i];
                });
            });
            

            sycl::buffer<float> diff_buf(n);
            
            queue.submit([&](sycl::handler& cgh) {
                auto x_old_acc = x_old_buf.get_access<sycl::access::mode::read>(cgh);
                auto x_new_acc = x_new_buf.get_access<sycl::access::mode::read>(cgh);
                auto diff_acc = diff_buf.get_access<sycl::access::mode::write>(cgh);
                
                cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    diff_acc[idx] = sycl::fabs(x_new_acc[idx] - x_old_acc[idx]);
                });
            });
            
            {
                auto diff_acc = diff_buf.get_host_access();
                max_diff = 0.0f;
                for (size_t i = 0; i < n; ++i) {
                    if (diff_acc[i] > max_diff) {
                        max_diff = diff_acc[i];
                    }
                }
            }
            
            queue.submit([&](sycl::handler& cgh) {
                auto x_old_acc = x_old_buf.get_access<sycl::access::mode::write>(cgh);
                auto x_new_acc = x_new_buf.get_access<sycl::access::mode::read>(cgh);
                
                cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    x_old_acc[idx] = x_new_acc[idx];
                });
            });
            
            iteration++;
            
        } while (iteration < ITERATIONS && max_diff >= accuracy);
        
        {
            auto x_new_acc = x_new_buf.get_host_access();
            for (size_t i = 0; i < n; ++i) {
                x[i] = x_new_acc[i];
            }
        }
        
    } catch (sycl::exception& e) {
        return std::vector<float>();
    }
    
    return x;
}