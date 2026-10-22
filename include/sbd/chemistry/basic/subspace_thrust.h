/**
@file sbd/chemistry/basic/subspace_thrust.h
@brief where the Davidson/Lanczos subspace vectors live: GPU (HBM) or CPU (LPDDR5X)
*/
#ifndef SBD_CHEMISTRY_SUBSPACE_THRUST_H
#define SBD_CHEMISTRY_SUBSPACE_THRUST_H

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

#include "sbd/framework/nvtx.h"
#include "sbd/framework/thrust_kernels.h"
#include "sbd/framework/dm_vector.h"
#include "sbd/framework/mpi_utility_thrust.h"

namespace sbd
{

// std::vector allocator whose resize leaves elements uninitialized, so the host subspace vectors
// (8 GB each at 1e9 determinants) can be zeroed, and their pages first touched, by all OpenMP
// threads instead of by one thread inside resize (about 135 s for 20 such vectors).
template <typename T>
struct default_init_allocator : std::allocator<T> {
    template <typename U> struct rebind { using other = default_init_allocator<U>; };
    using std::allocator<T>::allocator;
    default_init_allocator() = default;
    template <typename U> default_init_allocator(const default_init_allocator<U>&) noexcept {}
    template <typename U> void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value)
    {
        ::new (static_cast<void*>(p)) U;
    }
    template <typename U, typename... Args> void construct(U* p, Args&&... args)
    {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
};

// Subspace vectors in GPU memory; element-wise work runs on the GPU.
struct DeviceSubspace {
    template <typename T> using vector = thrust::device_vector<T>;
    static constexpr bool on_device = true;
#ifdef SBD_USE_THRUST_NOSYNC
    static auto policy() { return thrust::cuda::par_nosync.on(0); }
#else
    static auto policy() { return thrust::cuda::par.on(0); }
#endif
    static void sync() {
#ifdef SBD_USE_THRUST_NOSYNC
        cudaStreamSynchronize(0);
#endif
    }
    static void check(MPI_Comm) {}
    template <typename T> static void allocate(vector<T>& v, size_t n) { v.resize(n); }
};

// Subspace vectors in host memory; element-wise work runs on the CPU with OpenMP.
// Only the mult input/output is staged through GPU memory (SubspaceMult).
struct HostSubspace {
    template <typename T> using vector = std::vector<T, default_init_allocator<T>>;
    static constexpr bool on_device = false;
    static auto policy() { return thrust::omp::par; }
    static void sync() {}
    // Element-wise work and reductions on the subspace run on OpenMP threads.
    static void check(MPI_Comm comm) {
        if (omp_get_max_threads() == 1) {
            int rank; MPI_Comm_rank(comm, &rank);
            if (rank == 0)
                std::cerr << "Error: --cpu_subspace 1 needs OpenMP threads > 1. "
                             "Set OMP_NUM_THREADS (and srun --cpus-per-task).\n";
            MPI_Abort(comm, 1);
        }
    }
    // zero-filled like resize, but by all threads (parallel first touch)
    template <typename T> static void allocate(vector<T>& v, size_t n)
    {
        v.resize(n);
        T* p = v.data();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) p[i] = T(0);
    }
};

template <typename T>
inline T* raw_ptr(thrust::device_vector<T>& v) { return thrust::raw_pointer_cast(v.data()); }
template <typename T, typename Alloc>
inline T* raw_ptr(std::vector<T, Alloc>& v) { return v.data(); }

// Host counterpart of the device Normalize(X, res, comm, comm_size).
template <typename ElemT, typename RealT, typename Alloc>
void Normalize(std::vector<ElemT, Alloc>& X, RealT& res, MPI_Comm comm, int /*comm_size*/)
{
    SBD_NVTX_RANGE_COLOR("Normalize", __LINE__);
    Normalize(X, res, comm);
}

// Hamiltonian multiplication on subspace vectors: HC = H*C (zero) or HC += H*C (add).
// The device version calls mult.run directly; the host version stages C and HC
// through two GPU buffers, a fixed HBM cost independent of the subspace depth.
template <typename Space, typename ElemT>
class SubspaceMult;

template <typename ElemT>
class SubspaceMult<DeviceSubspace, ElemT> {
    MultBase<ElemT>& mult_;
    const thrust::device_vector<ElemT>& hii_;
public:
    SubspaceMult(MultBase<ElemT>& mult, const thrust::device_vector<ElemT>& hii, size_t)
        : mult_(mult), hii_(hii) {}
    void zero(thrust::device_vector<ElemT>& C, thrust::device_vector<ElemT>& HC) {
        thrust::fill(HC.begin(), HC.end(), 0);
        add(C, HC);
    }
    void add(thrust::device_vector<ElemT>& C, thrust::device_vector<ElemT>& HC) {
        SBD_NVTX_RANGE_COLOR("mult.run", __LINE__);
        mult_.run(hii_, C, HC);
    }
};

template <typename ElemT>
class SubspaceMult<HostSubspace, ElemT> {
    using Vector = HostSubspace::vector<ElemT>;
    MultBase<ElemT>& mult_;
    const thrust::device_vector<ElemT>& hii_;
    thrust::device_vector<ElemT> C_dev_;
    thrust::device_vector<ElemT> HC_dev_;
public:
    SubspaceMult(MultBase<ElemT>& mult, const thrust::device_vector<ElemT>& hii, size_t n)
        : mult_(mult), hii_(hii), C_dev_(n), HC_dev_(n) {}
    void zero(Vector& C, Vector& HC) {
        thrust::fill(HC_dev_.begin(), HC_dev_.end(), 0);
        run(C, HC);
    }
    void add(Vector& C, Vector& HC) {
        thrust::copy(HC.begin(), HC.end(), HC_dev_.begin());
        run(C, HC);
    }
private:
    void run(Vector& C, Vector& HC) {
        thrust::copy(C.begin(), C.end(), C_dev_.begin());
        {
            SBD_NVTX_RANGE_COLOR("mult.run", __LINE__);
            mult_.run(hii_, C_dev_, HC_dev_);
        }
        thrust::copy(HC_dev_.begin(), HC_dev_.end(), HC.begin());
    }
};

}

#endif
