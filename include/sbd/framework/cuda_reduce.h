// This is a part of qsbd
/**
@file /sbd/framework/cuda_reduce.h
@brief function for vector on distributed-memory
*/

#ifndef SBD_FRAMEWORK_REDUCE_CUDA_H
#define SBD_FRAMEWORK_REDUCE_CUDA_H

#include <cuda.h>
#include <complex>

#define _WS 32
#define _MAX_THD 1024

namespace sbd {

// ---------------------------------------------------------------------------
// First-pass kernel: user-supplied functor, one value per thread.
// Writes (sum, correction) pairs to pReduceBuffer[b*2 + 0..1].
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
    nw = min(blockDim.x, (unsigned)_WS);
    for (j = 1; j < nw; j *= 2) {
        c += __shfl_xor_sync(0xffffffff, c, j, 32);
        v = __shfl_xor_sync(0xffffffff, sum, j, 32) - c;
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
                c += __shfl_xor_sync(0xffffffff, c, j, 32);
                v = __shfl_xor_sync(0xffffffff, sum, j, 32) - c;
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
    nw = min(blockDim.x, (unsigned)_WS);
    for (j = 1; j < nw; j *= 2) {
        c += __shfl_xor_sync(0xffffffff, c, j, 32);
        v = __shfl_xor_sync(0xffffffff, sum, j, 32) - c;
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
                c += __shfl_xor_sync(0xffffffff, c, j, 32);
                v = __shfl_xor_sync(0xffffffff, sum, j, 32) - c;
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
// First-pass kernel for complex<double> inner product.
// Fetches X[i] and Y[i] once per thread, computes both components of
// conj(X[i]) * Y[i] = (xr*yr + xi*yi) + i*(xr*yi - xi*yr),
// and Kahan-accumulates re and im simultaneously.
// Writes 4 doubles per block: out[b*4+0..3] = (sum_re, c_re, sum_im, c_im).
// ---------------------------------------------------------------------------
template <typename ElemT>
__global__ void dev_precise_reduce_complex_ip(
    const ElemT* X, const ElemT* Y, double* out, size_t count)
{
    __shared__ double cache[_MAX_THD * 4 / _WS];
    double sum_re = 0.0, c_re = 0.0;
    double sum_im = 0.0, c_im = 0.0;
    double t, v;
    size_t j, nw;

    size_t i = threadIdx.x + blockIdx.x * (size_t)blockDim.x;
    if (i < count) {
        double xr = X[i].real(), xi = X[i].imag();
        double yr = Y[i].real(), yi = Y[i].imag();
        // Full TwoSum (Knuth/Ogita-Rump-Oishi): correct for any magnitude ordering.
        // s = fl(a+b); bb = fl(s-a) ≈ b; aa = fl(s-bb) ≈ a; err = (a-aa)+(b-bb).
        double prod_rr = xr * yr, prod_ii = xi * yi;
        sum_re = prod_rr + prod_ii;
        { double bb = sum_re - prod_rr; double aa = sum_re - bb;
          c_re = (prod_rr - aa) + (prod_ii - bb); }
        // For subtraction: treat as sum_im = prod_ri + (-prod_ir).
        double prod_ri = xr * yi, prod_ir = xi * yr;
        sum_im = prod_ri - prod_ir;
        { double bb = sum_im - prod_ri; double aa = sum_im - bb;
          c_im = (prod_ri - aa) + (-prod_ir - bb); }
    }

    nw = min(blockDim.x, (unsigned)_WS);
    for (j = 1; j < nw; j *= 2) {
        c_re  += __shfl_xor_sync(0xffffffff, c_re,  j, 32);
        v      = __shfl_xor_sync(0xffffffff, sum_re, j, 32) - c_re;
        t = sum_re + v; { double bb = t - sum_re; double aa = t - bb; c_re = (sum_re - aa) + (v - bb); } sum_re = t;
        c_im  += __shfl_xor_sync(0xffffffff, c_im,  j, 32);
        v      = __shfl_xor_sync(0xffffffff, sum_im, j, 32) - c_im;
        t = sum_im + v; { double bb = t - sum_im; double aa = t - bb; c_im = (sum_im - aa) + (v - bb); } sum_im = t;
    }

    if (blockDim.x > _WS) {
        int warp_id = (int)(threadIdx.x / _WS);
        int lane    = (int)(threadIdx.x & (_WS - 1));
        if (lane == 0) {
            cache[warp_id * 4 + 0] = sum_re; cache[warp_id * 4 + 1] = c_re;
            cache[warp_id * 4 + 2] = sum_im; cache[warp_id * 4 + 3] = c_im;
        }
        __syncthreads();
        int num_warps = ((int)blockDim.x + _WS - 1) / _WS;
        if (threadIdx.x < (unsigned)_WS) {
            if ((int)threadIdx.x < num_warps) {
                sum_re = cache[threadIdx.x * 4 + 0]; c_re = cache[threadIdx.x * 4 + 1];
                sum_im = cache[threadIdx.x * 4 + 2]; c_im = cache[threadIdx.x * 4 + 3];
            } else {
                sum_re = c_re = sum_im = c_im = 0.0;
            }
            nw = _WS;
            for (j = 1; j < nw; j *= 2) {
                c_re  += __shfl_xor_sync(0xffffffff, c_re,  j, 32);
                v      = __shfl_xor_sync(0xffffffff, sum_re, j, 32) - c_re;
                t = sum_re + v; { double bb = t - sum_re; double aa = t - bb; c_re = (sum_re - aa) + (v - bb); } sum_re = t;
                c_im  += __shfl_xor_sync(0xffffffff, c_im,  j, 32);
                v      = __shfl_xor_sync(0xffffffff, sum_im, j, 32) - c_im;
                t = sum_im + v; { double bb = t - sum_im; double aa = t - bb; c_im = (sum_im - aa) + (v - bb); } sum_im = t;
            }
        }
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x * 4 + 0] = sum_re;
        out[blockIdx.x * 4 + 1] = c_re;
        out[blockIdx.x * 4 + 2] = sum_im;
        out[blockIdx.x * 4 + 3] = c_im;
    }
}

// ---------------------------------------------------------------------------
// Race-free two-pointer second-pass kernel (2-component).
// Reads 4 doubles per slot from in[i*4 + 0..3]; writes per block to out[b*4 + 0..3].
// 'in' and 'out' must not alias.  count = number of 4-double slots in 'in'.
// ---------------------------------------------------------------------------
__global__ void dev_precise_reduce2_step(const double* in, double* out, size_t count)
{
    __shared__ double cache[_MAX_THD * 4 / _WS];
    double sum_re, c_re, sum_im, c_im;
    double t, v;
    size_t j, nw;

    size_t i = threadIdx.x + blockIdx.x * (size_t)blockDim.x;
    if (i < count) {
        sum_re = in[i * 4 + 0]; c_re = in[i * 4 + 1];
        sum_im = in[i * 4 + 2]; c_im = in[i * 4 + 3];
    } else {
        sum_re = c_re = sum_im = c_im = 0.0;
    }

    nw = min(blockDim.x, (unsigned)_WS);
    for (j = 1; j < nw; j *= 2) {
        c_re  += __shfl_xor_sync(0xffffffff, c_re,  j, 32);
        v      = __shfl_xor_sync(0xffffffff, sum_re, j, 32) - c_re;
        t = sum_re + v; { double bb = t - sum_re; double aa = t - bb; c_re = (sum_re - aa) + (v - bb); } sum_re = t;
        c_im  += __shfl_xor_sync(0xffffffff, c_im,  j, 32);
        v      = __shfl_xor_sync(0xffffffff, sum_im, j, 32) - c_im;
        t = sum_im + v; { double bb = t - sum_im; double aa = t - bb; c_im = (sum_im - aa) + (v - bb); } sum_im = t;
    }

    if (blockDim.x > _WS) {
        int warp_id = (int)(threadIdx.x / _WS);
        int lane    = (int)(threadIdx.x & (_WS - 1));
        if (lane == 0) {
            cache[warp_id * 4 + 0] = sum_re; cache[warp_id * 4 + 1] = c_re;
            cache[warp_id * 4 + 2] = sum_im; cache[warp_id * 4 + 3] = c_im;
        }
        __syncthreads();
        int num_warps = ((int)blockDim.x + _WS - 1) / _WS;
        if (threadIdx.x < (unsigned)_WS) {
            if ((int)threadIdx.x < num_warps) {
                sum_re = cache[threadIdx.x * 4 + 0]; c_re = cache[threadIdx.x * 4 + 1];
                sum_im = cache[threadIdx.x * 4 + 2]; c_im = cache[threadIdx.x * 4 + 3];
            } else {
                sum_re = c_re = sum_im = c_im = 0.0;
            }
            nw = _WS;
            for (j = 1; j < nw; j *= 2) {
                c_re  += __shfl_xor_sync(0xffffffff, c_re,  j, 32);
                v      = __shfl_xor_sync(0xffffffff, sum_re, j, 32) - c_re;
                t = sum_re + v; { double bb = t - sum_re; double aa = t - bb; c_re = (sum_re - aa) + (v - bb); } sum_re = t;
                c_im  += __shfl_xor_sync(0xffffffff, c_im,  j, 32);
                v      = __shfl_xor_sync(0xffffffff, sum_im, j, 32) - c_im;
                t = sum_im + v; { double bb = t - sum_im; double aa = t - bb; c_im = (sum_im - aa) + (v - bb); } sum_im = t;
            }
        }
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x * 4 + 0] = sum_re;
        out[blockIdx.x * 4 + 1] = c_re;
        out[blockIdx.x * 4 + 2] = sum_im;
        out[blockIdx.x * 4 + 3] = c_im;
    }
}

// ---------------------------------------------------------------------------
// Persistent device + pinned-host buffers shared across all instantiations.
// Grow-only, never freed.  Each TU gets its own instance (static linkage).
// ---------------------------------------------------------------------------
namespace detail {
    static double*  g_dev_buf  = nullptr;
    static size_t   g_dev_cap  = 0;         // capacity in doubles
    static double*  g_host_buf = nullptr;   // pinned host buffer, 8 doubles

    inline double* ensure_reduce_buf(size_t n_doubles) {
        if (n_doubles > g_dev_cap) {
            cudaFree(g_dev_buf);            // no-op if nullptr
            cudaMalloc(&g_dev_buf, n_doubles * sizeof(double));
            g_dev_cap = n_doubles;
        }
        if (!g_host_buf) {
            cudaMallocHost(&g_host_buf, 8 * sizeof(double));
        }
        return g_dev_buf;
    }
}

// ---------------------------------------------------------------------------
// Updated precise_reduce_sum_with_function:
//   - Persistent device buffer (no per-call cudaMalloc/cudaFree)
//   - Race-free second pass via dev_precise_reduce_step (separate in/out)
//   - Single async D→H copy + stream sync instead of two synchronous copies
// ---------------------------------------------------------------------------
template <typename Function>
double precise_reduce_sum_with_function(Function func, size_t size)
{
    // A zero-length reduction sums to 0. Returning early avoids launching the
    // reduce kernel with a zero block dimension (<<<1, 0>>>): that launch fails
    // and leaves a pending CUDA error for the next Thrust dispatch to trip over.
    if (size == 0) return 0.0;
    SBD_NVTX_RANGE_COLOR("precise_reduce_sum_with_function", __LINE__);

    size_t nt = size, nb = 1;
    if (nt > _MAX_THD) {
        nb = (nt + _MAX_THD - 1) / _MAX_THD;
        nt = _MAX_THD;
    }

    // Total buffer needed: first pass (nb*2) + all subsequent passes (< nb*2 combined).
    // 3*nb+8 is always sufficient; grows on the first call for a larger input.
    double* buf = detail::ensure_reduce_buf(3 * nb + 8);

    size_t src_off = 0;       // current pass reads from buf + src_off
    size_t dst_off = nb * 2;  // current pass writes to  buf + dst_off

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

    cudaMemcpyAsync(detail::g_host_buf, buf + src_off, 2 * sizeof(double),
                    cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);
    return detail::g_host_buf[0];
}

// ---------------------------------------------------------------------------
// Fused complex inner product reduction.
// Computes Σ conj(X[i]) * Y[i] = (re, im) in two simultaneous Kahan sums.
// Single kernel pass over X and Y (vs 4 passes for the split real/imag approach).
// ElemT must provide .real() and .imag() accessible on device (e.g. std::complex<double>).
// ---------------------------------------------------------------------------
template <typename ElemT>
void precise_reduce_sum_complex_ip(
    const ElemT* X, const ElemT* Y, size_t size,
    double& re, double& im)
{
    if (size == 0) { re = im = 0.0; return; }

    size_t nt = size, nb = 1;
    if (nt > _MAX_THD) {
        nb = (nt + _MAX_THD - 1) / _MAX_THD;
        nt = _MAX_THD;
    }

    // Each block slot is 4 doubles.  5*nb+8 is always sufficient.
    double* buf = detail::ensure_reduce_buf(5 * nb + 8);

    size_t src_off = 0;
    size_t dst_off = nb * 4;

    dev_precise_reduce_complex_ip<ElemT><<<nb, nt>>>(X, Y, buf, size);

    while (nb > 1) {
        size_t n = nb;
        nt = nb; nb = 1;
        if (nt > _MAX_THD) {
            nb = (nt + _MAX_THD - 1) / _MAX_THD;
            nt = _MAX_THD;
        }
        dev_precise_reduce2_step<<<nb, nt>>>(buf + src_off, buf + dst_off, n);
        src_off  = dst_off;
        dst_off += nb * 4;
    }

    cudaMemcpyAsync(detail::g_host_buf, buf + src_off, 4 * sizeof(double),
                    cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);
    re = detail::g_host_buf[0];   // g_host_buf[1] = c_re (discarded)
    im = detail::g_host_buf[2];   // g_host_buf[3] = c_im (discarded)
}


}

#endif
