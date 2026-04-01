/**
@file sbd/framework/thrust_kernels.h
@brief kernel classes for Thrust
*/
#ifndef SBD_FRAMEWORK_THRUST_KERNELS_H
#define SBD_FRAMEWORK_THRUST_KERNELS_H

#include "sbd/framework/nvtx.h"

#ifdef SBD_USE_CUBLAS
#include "sbd/framework/cublas_utility.h"
#endif

namespace sbd
{

// AXPY kernel
template <typename ElemT>
struct AXPY_kernel {
    ElemT a;

    AXPY_kernel(ElemT a_in) : a(a_in) {}

    __host__ __device__ ElemT operator()(const ElemT& x, const ElemT& y) const
    {
        return a * x + y;
    }
};

// AX kernel
template <typename ElemT>
struct AX_kernel {
    ElemT a;

    AX_kernel(ElemT a_in) : a(a_in) {}

    __host__ __device__ ElemT operator()(const ElemT& x) const
    {
        return a * x;
    }
};


// dot product
template <typename ElemT>
struct dot_product_kernel {
    ElemT* A;
    ElemT* B;

    dot_product_kernel(const thrust::device_vector<ElemT>& a, const thrust::device_vector<ElemT>& b)
    {
        A = (ElemT*)thrust::raw_pointer_cast(a.data());
        B = (ElemT*)thrust::raw_pointer_cast(b.data());
    }

    __host__ __device__ ElemT operator()(const size_t i) const
    {
        return A[i] * B[i];
    }
};

template <typename ElemT, typename RealT>
void Normalize(thrust::device_vector<ElemT>& X,
               RealT& res,
               MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("Normalize", __LINE__);

    res = 0.0;
    RealT sum = 0.0;

    /*
    // If CUDA native kernel can not be used, use host code
    std::vector<ElemT> hx(X.size());
    thrust::copy_n(X.begin(), X.size(), hx.begin());
    Normalize(hx, res, comm);
    thrust::copy_n(hx.begin(), hx.size(), X.begin());
    */

    auto kernel = dot_product_kernel<RealT>(X, X);
    sum = precise_reduce_sum_with_function(kernel, X.size());

    MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Allreduce(&sum, &res, 1, DataT, MPI_SUM, comm);
    }
    res = std::sqrt(res);
    ElemT factor = ElemT(1.0 / res);

    {
        SBD_NVTX_RANGE_COLOR("thrust::transform", __LINE__);
        // thrust::transform(thrust::device, X.begin(), X.end(), thrust::constant_iterator<ElemT>(factor), X.begin(), thrust::multiplies<ElemT>());
        thrust::transform(thrust::device, X.begin(), X.end(), X.begin(), AX_kernel<ElemT>(factor));
    }
}

template <typename ElemT, typename RealT>
void InnerProduct(const thrust::device_vector<ElemT>& X,
                  const thrust::device_vector<ElemT>& Y,
                  RealT& res,
                  MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("InnerProduct", __LINE__);

    res = 0.0;
    RealT sum = 0.0;

    /*
    // If CUDA native kernel can not be used, use host code
    std::vector<ElemT> hx(X.size());
    thrust::copy_n(X.begin(), X.size(), hx.begin());
    std::vector<ElemT> hy(Y.size());
    thrust::copy_n(Y.begin(), Y.size(), hy.begin());
    InnerProduct(hx, hy, res, comm);
    */

#ifdef SBD_USE_CUBLAS
    sum = sbd::Dot(X, Y);
#else
    auto kernel = dot_product_kernel<RealT>(X, Y);
    sum = precise_reduce_sum_with_function(kernel, X.size());
#endif
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
        MPI_Allreduce(&sum, &res, 1, DataT, MPI_SUM, comm);
    }
}

template <typename ElemT, typename RealT>
void BatchedInnerProduct(std::vector<thrust::device_vector<ElemT>>& X,
                         int num_X,
                         const thrust::device_vector<ElemT>& Y,
                         std::vector<RealT>& Z,
                         MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("BatchedInnerProduct", __LINE__);
    if (num_X == 0) return;
    assert(num_X <= X.size());
    assert(num_X <= Z.size());
#ifndef SBD_USE_CUBLAS
    for (int i = 0; i < num_X; i++) {
        InnerProduct(X[i], Y, Z[i], comm);
    }
#else // #ifndef SBD_USE_CUBLAS
    for (int i = 0; i < num_X; i++) {
        res[i] = sbd::Dot(X[i], Y);
    }
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
        MPI_Allreduce(res.data(), Z.data(), num_X, DataT, MPI_SUM, comm);
    }
#endif // #ifndef SBD_USE_CUBLAS
}

#ifdef SBD_USE_CUBLAS
template <typename ElemT, typename RealT>
void BatchedInnerProduct_GEMV(std::vector<thrust::device_vector<ElemT>>& X,
                              int num_X,
                              const thrust::device_vector<ElemT>& Y,
                              std::vector<RealT>& Z,
                              MPI_Comm comm,
                              ElemT* ws_ptr = nullptr)
{
    SBD_NVTX_RANGE_COLOR("BatchedInnerProduct_GEMV", __LINE__);
    if (num_X == 0) return;

    static_assert(std::is_same<ElemT, RealT>::value, "ElemT must match RealT");
    assert(num_X <= X.size());
    assert(num_X <= Z.size());

    cudaStream_t stream = 0;
    const size_t N = Y.size();
    thrust::device_vector<ElemT> X_flat;
    thrust::device_vector<ElemT> Z_dev;
    ElemT *X_flat_ptr;
    ElemT *Z_dev_ptr;
    if (ws_ptr == nullptr) {
        X_flat.resize(num_X * N);
        Z_dev.resize(num_X);
        X_flat_ptr = thrust::raw_pointer_cast(X_flat.data());
        Z_dev_ptr = thrust::raw_pointer_cast(Z_dev.data());
    } else {
        X_flat_ptr = ws_ptr;
        Z_dev_ptr = ws_ptr + (N * num_X);
    }
    for (int i = 0; i < num_X; i++) {
        assert(X[i].size() == N);
        cudaMemcpyAsync(
            X_flat_ptr + (N * i),
            thrust::raw_pointer_cast(X[i].data()),
            sizeof(ElemT) * N,
            cudaMemcpyDeviceToDevice, stream);
    }
    auto& ctx = sbd::GetCublasContext();
    ctx.set_pointer_mode_host();
    ctx.set_stream(stream);
    const ElemT alpha = static_cast<ElemT>(1.0);
    const ElemT beta  = static_cast<ElemT>(0.0);
    const ElemT* A = X_flat_ptr;
    const ElemT* x = thrust::raw_pointer_cast(Y.data());
    ElemT* y       = Z_dev_ptr;
    if constexpr (std::is_same<ElemT, float>::value) {
        sbd::CheckCublas(
            cublasSgemv(ctx.get(), CUBLAS_OP_T, N, num_X,
                        &alpha, A, N, x, 1, &beta, y, 1),
            "cublasSgemv failed");
    }
    else if constexpr (std::is_same<ElemT, double>::value) {
        sbd::CheckCublas(
            cublasDgemv(ctx.get(),CUBLAS_OP_T, N, num_X,
                        &alpha, A, N, x, 1, &beta, y, 1),
            "cublasDgemv failed");
    }
    else {
        static_assert(!sizeof(ElemT), "Unsupported type for GEMV");
    }
    std::vector<ElemT> res(num_X);
    cudaMemcpyAsync(res.data(), Z_dev_ptr, sizeof(ElemT) * num_X,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
        MPI_Allreduce(res.data(), Z.data(), num_X, DataT, MPI_SUM, comm);
    }
}
#endif

}

#endif
