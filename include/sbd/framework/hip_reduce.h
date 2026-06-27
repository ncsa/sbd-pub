#include "hip/hip_runtime.h"
// This is a part of qsbd
/**
@file /sbd/framework/hip_reduce.h
@brief function for vector on distributed-memory
*/

#ifndef SBD_FRAMEWORK_REDUCE_HIP_H
#define SBD_FRAMEWORK_REDUCE_HIP_H

#include <hip/hip_runtime.h>

#define _WS 64
#define _MAX_THD 1024
namespace sbd {

// ---------------------------------------------------------------------------
// First-pass kernel: user-supplied functor, one value per thread.
// ---------------------------------------------------------------------------
template <typename kernel_t>
__global__ void dev_precise_reduce_with_function(double *pReduceBuffer, kernel_t func, size_t count)
{
    __shared__ double cache[_MAX_THD * 2 / _WS];
    double sum, t, v;
    double c = 0.0;
    size_t i, j, nw;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= count)
        sum = 0.0;
    else
        sum = func(i);

    // reduce in warp
    nw = min(blockDim.x, _WS);
    for (j = 1; j < nw; j *= 2) {
        c += __shfl_xor(c, j, 64);
        v = __shfl_xor(sum, j, 64) - c;
        t = sum + v;
        { double bb = t - sum; double aa = t - bb; c = (sum - aa) + (v - bb); }
        sum = t;
    }

    if (blockDim.x > _WS) {
        // reduce in thread block
        if ((threadIdx.x & (_WS - 1)) == 0) {
            cache[(threadIdx.x / _WS) * 2] = sum;
            cache[(threadIdx.x / _WS) * 2 + 1] = c;
        }
        __syncthreads();
        if (threadIdx.x < _WS) {
            if (threadIdx.x < ((blockDim.x + _WS - 1) / _WS)) {
                sum = cache[threadIdx.x*2];
                c = cache[threadIdx.x*2 + 1];
            } else {
                sum = 0.0;
                c = 0.0;
            }

            // reduce in warp
            nw = _WS;
            for (j = 1; j < nw; j *= 2) {
                c += __shfl_xor(c, j, 64);
                v = __shfl_xor(sum, j, 64) - c;
                t = sum + v;
                { double bb = t - sum; double aa = t - bb; c = (sum - aa) + (v - bb); }
                sum = t;
            }
        }
    }

    if (threadIdx.x == 0) {
        pReduceBuffer[blockIdx.x * 2] = sum;
        pReduceBuffer[blockIdx.x * 2 + 1] = c;
    }
}

// ---------------------------------------------------------------------------
// Race-free two-pointer second-pass kernel (1-component).
// Reads (sum, c) pairs from in[i*2 + 0..1]; writes per block to out[b*2 + 0..1].
// 'in' and 'out' must not alias.  count = number of pairs in 'in'.
// ---------------------------------------------------------------------------
__global__ void dev_precise_reduce_step(const double* in, double* out, size_t count)
{
    __shared__ double cache[_MAX_THD * 2 / _WS];
    double sum, t, v;
    double c = 0.0;
    size_t i, j, nw;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= count)
        sum = 0.0;
    else{
        sum = in[i*2];
        c = in[i*2 + 1];
    }

    // reduce in warp
    nw = min(blockDim.x, _WS);
    for (j = 1; j < nw; j *= 2) {
        c += __shfl_xor(c, j, 64);
        v = __shfl_xor(sum, j, 64) - c;
        t = sum + v;
        { double bb = t - sum; double aa = t - bb; c = (sum - aa) + (v - bb); }
        sum = t;
    }

    if (blockDim.x > _WS) {
        // reduce in thread block
        if ((threadIdx.x & (_WS - 1)) == 0) {
            cache[(threadIdx.x / _WS) * 2] = sum;
            cache[(threadIdx.x / _WS) * 2 + 1] = c;
        }
        __syncthreads();
        if (threadIdx.x < _WS) {
            if (threadIdx.x < ((blockDim.x + _WS - 1) / _WS)) {
                sum = cache[threadIdx.x*2];
                c = cache[threadIdx.x*2 + 1];
            } else {
                sum = 0.0;
                c = 0.0;
            }

            // reduce in warp
            nw = _WS;
            for (j = 1; j < nw; j *= 2) {
                c += __shfl_xor(c, j, 64);
                v = __shfl_xor(sum, j, 64) - c;
                t = sum + v;
                { double bb = t - sum; double aa = t - bb; c = (sum - aa) + (v - bb); }
                sum = t;
            }
        }
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x * 2] = sum;
        out[blockIdx.x * 2 + 1] = c;
    }
}

// ---------------------------------------------------------------------------
// Persistent device + pinned-host buffers.  Grow-only, never freed.
// ---------------------------------------------------------------------------
namespace detail {
    static double*  g_dev_buf  = nullptr;
    static size_t   g_dev_cap  = 0;
    static double*  g_host_buf = nullptr;

    inline double* ensure_reduce_buf(size_t n_doubles) {
        if (n_doubles > g_dev_cap) {
            hipFree(g_dev_buf);
            hipMalloc(&g_dev_buf, n_doubles * sizeof(double));
            g_dev_cap = n_doubles;
        }
        if (!g_host_buf) {
            hipHostMalloc(&g_host_buf, 8 * sizeof(double), 0);
        }
        return g_dev_buf;
    }
}

// ---------------------------------------------------------------------------
// Updated precise_reduce_sum_with_function: persistent buffer, race-free step
// kernel, single async D→H copy.
// ---------------------------------------------------------------------------
template <typename Function>
double precise_reduce_sum_with_function(Function func, size_t size)
{
    if (size == 0) return 0.0;

    size_t nt = size, nb = 1;
    if (nt > _MAX_THD) {
        nb = (nt + _MAX_THD - 1) / _MAX_THD;
        nt = _MAX_THD;
    }

    double* buf = detail::ensure_reduce_buf(3 * nb + 8);

    size_t src_off = 0;
    size_t dst_off = nb * 2;

    dev_precise_reduce_with_function<Function><<<nb, nt>>>(buf, func, size);

    while (nb > 1) {
        size_t n = nb;
        nt = nb; nb = 1;
        if (nt > _MAX_THD) {
            nb = (nt + _MAX_THD - 1) / _MAX_THD;
            nt = _MAX_THD;
        }
        dev_precise_reduce_step<<<nb, nt>>>(buf + src_off, buf + dst_off, n);
        src_off  = dst_off;
        dst_off += nb * 2;
    }

    hipMemcpyAsync(detail::g_host_buf, buf + src_off, 2 * sizeof(double),
                   hipMemcpyDeviceToHost, 0);
    hipStreamSynchronize(0);
    return detail::g_host_buf[0];
}


}

#endif
