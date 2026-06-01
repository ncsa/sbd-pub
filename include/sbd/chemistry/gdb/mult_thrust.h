/**
@file sbd/chemistry/tpb/mult.h
@brief Function to perform Hamiltonian operation for general determinant basis
*/
#ifndef SBD_CHEMISTRY_GDB_MULT_THRUST_H
#define SBD_CHEMISTRY_GDB_MULT_THRUST_H


#include "sbd/framework/mpi_utility_thrust.h"
#include <cassert>
#include <chrono>

// SUBWARP threading for MultAlphaBetaKernel (track gh-A).
// SBD_GDB_SUBWARP_SIZE selects the threading granularity:
//   1            → stride=1, behaves like the original code path
//                  (cub::WarpReduce<T,1> is identity; no cross-lane comm)
//   2 / 4 / 8 / 16 / 32 → strided inner-k loop + cub::WarpReduce reduction
//                  + per-iteration ballot-based candidate buffer.
// Single code path for all sizes — no #ifdef'd duplicate body.
#ifndef SBD_GDB_SUBWARP_SIZE
  #define SBD_GDB_SUBWARP_SIZE 8
#endif
static_assert(SBD_GDB_SUBWARP_SIZE == 1  ||
              SBD_GDB_SUBWARP_SIZE == 2  ||
              SBD_GDB_SUBWARP_SIZE == 4  ||
              SBD_GDB_SUBWARP_SIZE == 8  ||
              SBD_GDB_SUBWARP_SIZE == 16 ||
              SBD_GDB_SUBWARP_SIZE == 32,
              "SBD_GDB_SUBWARP_SIZE must be 1, 2, 4, 8, 16, or 32");

// SBD_GDB_AB_SUBWARP_SIZE overrides SubwarpSize for MultAlphaBetaKernel only.
// When unset, defaults to SBD_GDB_SUBWARP_SIZE (uniform subwarp granularity
// across all five mult kernels). Set independently to sweep the AlphaBeta
// subwarp size without changing the other four kernels.
#ifndef SBD_GDB_AB_SUBWARP_SIZE
  #define SBD_GDB_AB_SUBWARP_SIZE 32
#endif
static_assert(SBD_GDB_AB_SUBWARP_SIZE == 1  ||
              SBD_GDB_AB_SUBWARP_SIZE == 2  ||
              SBD_GDB_AB_SUBWARP_SIZE == 4  ||
              SBD_GDB_AB_SUBWARP_SIZE == 8  ||
              SBD_GDB_AB_SUBWARP_SIZE == 16 ||
              SBD_GDB_AB_SUBWARP_SIZE == 32,
              "SBD_GDB_AB_SUBWARP_SIZE must be 1, 2, 4, 8, 16, or 32");

#include <cub/warp/warp_reduce.cuh>

namespace sbd
{
namespace gdb
{

#ifndef SBD_MULT_BLOCK_SIZE
  #define SBD_MULT_BLOCK_SIZE 64
#endif
static_assert(SBD_MULT_BLOCK_SIZE == 32  ||
              SBD_MULT_BLOCK_SIZE == 64  ||
              SBD_MULT_BLOCK_SIZE == 128 ||
              SBD_MULT_BLOCK_SIZE == 256,
              "SBD_MULT_BLOCK_SIZE must be 32, 64, 128, or 256");
static_assert(SBD_MULT_BLOCK_SIZE % SBD_GDB_SUBWARP_SIZE == 0,
              "SBD_MULT_BLOCK_SIZE must be a multiple of SBD_GDB_SUBWARP_SIZE.");
static_assert(SBD_MULT_BLOCK_SIZE % SBD_GDB_AB_SUBWARP_SIZE == 0,
              "SBD_MULT_BLOCK_SIZE must be a multiple of SBD_GDB_AB_SUBWARP_SIZE.");

// SBD_MULT_MIN_BLOCKS_PER_SM tunes the second argument of
// __launch_bounds__ on mult_for_each_n_kernel — the
// minBlocksPerMultiprocessor hint the compiler uses to pick a register
// budget per thread. Lower values trade occupancy for a larger per-thread
// register budget.
//   threads/SM = SBD_MULT_BLOCK_SIZE × SBD_MULT_MIN_BLOCKS_PER_SM
//   regs/thread budget ≈ 65536 / threads/SM
// Examples (BS=64):
//   mbpsm=32 → 2048 threads/SM (100% occupancy); 32 regs/thread
//   mbpsm=24 → 1536 threads/SM (75% occupancy); 42 regs/thread  ← default (track H)
//   mbpsm=20 → 1280 threads/SM (62.5% occupancy); 51 regs/thread
//   mbpsm=16 → 1024 threads/SM (50% occupancy); 64 regs/thread
#ifndef SBD_MULT_MIN_BLOCKS_PER_SM
  #define SBD_MULT_MIN_BLOCKS_PER_SM 24
#endif
static_assert(SBD_MULT_MIN_BLOCKS_PER_SM >= 1 &&
              SBD_MULT_MIN_BLOCKS_PER_SM <= 32,
              "SBD_MULT_MIN_BLOCKS_PER_SM must be in [1, 32] (H100 hw max).");

// Custom blocksize-tunable launcher for the GDB Mult kernels. Replaces
// thrust::for_each_n at the run()-level call sites with a fixed-block
// launch so the v2 SUBWARP-parallel kernel's shared-memory sizing
// (`BUF_TOTAL = 2 * SBD_MULT_BLOCK_SIZE`) matches the actual block size.
// BlockSize and MinBlocksPerSM are read from the Functor class (static constexpr
// members defined in MultKernelBase); n is the number of subwarps, not threads.
template <typename Functor>
__global__ __launch_bounds__(Functor::BlockSize, Functor::MinBlocksPerSM)
void mult_for_each_n_kernel(size_t n, Functor functor)
{
    size_t i = static_cast<size_t>(blockIdx.x) * Functor::BlockSize + threadIdx.x;
    if (i < n) functor(i);
}

template <typename Functor>
inline void launch_mult_for_each_n(size_t n_subwarps, Functor functor,
                                    cudaStream_t stream = 0)
{
    if (n_subwarps == 0) return;
    constexpr int BS = Functor::BlockSize;
    constexpr int SW = Functor::SubwarpSize;
    const size_t n_threads = n_subwarps * SW;
    const size_t grid = (n_threads + BS - 1) / BS;
    mult_for_each_n_kernel<Functor><<<grid, BS, 0, stream>>>(n_threads, functor);
}

template <typename ElemT>
class MultGDBThrust : public sbd::MultBase<ElemT> {
protected:
	thrust::device_vector<uint32_t> idxmap_storage;
	DetIndexMapThrust idxmap;
    thrust::device_vector<size_t> dets_;
    thrust::device_vector<size_t> detsums_;
    ElemT I0_;
    oneInt_Thrust<ElemT> I1_;
    thrust::device_vector<ElemT> I1_store;
    twoInt_Thrust<ElemT> I2_;
    thrust::device_vector<ElemT> I2_store;
    thrust::device_vector<ElemT> I2_dm;
    thrust::device_vector<ElemT> I2_em;

	std::vector<thrust::device_vector<uint32_t>> exidx_storage;
    std::vector<thrust::device_vector<int>> CrAn_storage;
	std::vector<ExcitationLookupThrust> exidx;
public:

	MultGDBThrust() {}

	const ElemT& I0(void) const
	{
		return I0_;
	}
	const oneInt_Thrust<ElemT>& I1(void) const
	{
		return I1_;
	}
	const twoInt_Thrust<ElemT>& I2(void) const
	{
		return I2_;
	}
	const thrust::device_vector<size_t>& dets(void) const
	{
		return dets_;
	}
	const thrust::device_vector<size_t>& detsums(void) const
	{
		return detsums_;
	}

	void Init(
        const size_t bit_length_in,
        const size_t norbs_in,
		const std::vector<std::vector<size_t>> &dets_in,
		const DetIndexMap &idxmap_in,
		const std::vector<ExcitationLookup>& exidx_in,
		const ElemT &I0_in,
        const oneInt<ElemT> &I1_in,
        const twoInt<ElemT> &I2_in,
		MPI_Comm h_comm,
		MPI_Comm b_comm,
		MPI_Comm t_comm);

    void run(const thrust::device_vector<ElemT> &hii,
                    const thrust::device_vector<ElemT> &Wk,
                    thrust::device_vector<ElemT> &Wb) override;

    void makeQChamDiagTerms(thrust::device_vector<ElemT> &hii);

	void correlation(const std::vector<ElemT> & w,
				std::vector<std::vector<ElemT>> & onebody_out,
				std::vector<std::vector<ElemT>> & twobody_out);
};

// contructor for Mult data
template <typename ElemT>
void MultGDBThrust<ElemT>::Init(
	    const size_t bit_length_in,
        const size_t norbs_in,
		const std::vector<std::vector<size_t>> &dets_in,
		const DetIndexMap &idxmap_in,
		const std::vector<ExcitationLookup>& exidx_in,
		const ElemT &I0_in,
        const oneInt<ElemT> &I1_in,
        const twoInt<ElemT> &I2_in,
		MPI_Comm h_comm_in,
		MPI_Comm b_comm_in,
		MPI_Comm t_comm_in)
{
    this->bit_length_ = bit_length_in;
    this->norbs_ = norbs_in;
    this->D_size_ = (2 * norbs_in + bit_length_in - 1) / bit_length_in;

    this->h_comm_ = h_comm_in;
    this->b_comm_ = b_comm_in;
    this->t_comm_ = t_comm_in;

    I0_ = I0_in;
    // copyin I1
    I1_store.resize(I1_in.store.size());
    thrust::copy_n(I1_in.store.begin(), I1_in.store.size(), I1_store.begin());
    I1_ = oneInt_Thrust<ElemT>(I1_store, I1_in.norbs);

    // copyin I2
    I2_store.resize(I2_in.store.size());
    thrust::copy_n(I2_in.store.begin(), I2_in.store.size(), I2_store.begin());
    I2_dm.resize(I2_in.DirectMat.size());
    thrust::copy_n(I2_in.DirectMat.begin(), I2_in.DirectMat.size(), I2_dm.begin());
    I2_em.resize(I2_in.ExchangeMat.size());
    thrust::copy_n(I2_in.ExchangeMat.begin(), I2_in.ExchangeMat.size(), I2_em.begin());
    I2_ = twoInt_Thrust<ElemT>(I2_store, I2_in.norbs, I2_dm, I2_em, I2_in.zero, I2_in.maxEntry);

    // copyin exidx
    exidx.clear();
    exidx_storage.resize(exidx_in.size());
    CrAn_storage.resize(exidx_in.size());
    for (size_t task = 0; task < exidx_in.size(); task++) {
        exidx.push_back(ExcitationLookupThrust(exidx_storage[task], CrAn_storage[task], exidx_in[task]));
    }

    // copyin dets
    dets_.resize(this->D_size_ * dets_in.size());
    for (int i = 0; i < dets_in.size(); i++) {
        thrust::copy_n(dets_in[i].begin(), this->D_size_, dets_.begin() + i * this->D_size_);
    }

    // compute detsums_: inclusive prefix parity of each determinant's bit string.
    // detsums_[d*D_size+w] bit k = parity of bits [0, w*bit_length+k] of det d.
    // Computed via a 6-step parallel prefix XOR scan per word, carrying parity
    // between words. Used by parity_fast() / TwoExciteFast() for O(1) parity lookup.
    {
        const size_t n_dets = dets_in.size();
        const size_t bl     = bit_length_in;
        std::vector<size_t> detsums_host(this->D_size_ * n_dets);
        for (size_t d = 0; d < n_dets; d++) {
            size_t carry = 0; // parity of all bits in preceding words (0 or 1)
            for (size_t w = 0; w < this->D_size_; w++) {
                size_t q = dets_in[d][w];
                // inclusive prefix XOR scan: q[k] = XOR of original bits [0..k]
                q ^= (q << 1);
                q ^= (q << 2);
                q ^= (q << 4);
                q ^= (q << 8);
                q ^= (q << 16);
                q ^= (q << 32);
                // apply carry from previous words
                if (carry) q = ~q;
                detsums_host[d * this->D_size_ + w] = q;
                carry = (q >> (bl - 1)) & 1; // parity of bits [0, (w+1)*bl-1]
            }
        }
        detsums_.resize(this->D_size_ * n_dets);
        thrust::copy_n(detsums_host.begin(), detsums_host.size(), detsums_.begin());
    }

	// copy indexmap
	idxmap = DetIndexMapThrust(idxmap_storage, idxmap_in);
}

template <typename ElemT>
class MultKernelBase : public DeterminantKernels<ElemT> {
public:
    static constexpr int BlockSize      = SBD_MULT_BLOCK_SIZE;
    static constexpr int SubwarpSize    = SBD_GDB_SUBWARP_SIZE;
    static constexpr int MinBlocksPerSM = SBD_MULT_MIN_BLOCKS_PER_SM;
protected:
    ElemT *wb;
    ElemT* twk;
    size_t* det;
    size_t* detsum;
public:
    MultKernelBase() {}

    MultKernelBase( const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data
                ) : DeterminantKernels<ElemT>(data.bit_length(), data.norbs(), data.I0(), data.I1(), data.I2())
    {
        wb = (ElemT*)thrust::raw_pointer_cast(v_wb.data());
        twk = (ElemT*)thrust::raw_pointer_cast(v_t.data());
        det    = (size_t*)thrust::raw_pointer_cast(data.dets().data());
        detsum = (size_t*)thrust::raw_pointer_cast(data.detsums().data());
    }

    MultKernelBase(const MultGDBThrust<ElemT>& data)
                 : DeterminantKernels<ElemT>(data.bit_length(), data.norbs(), data.I0(), data.I1(), data.I2())
    {
        det    = (size_t*)thrust::raw_pointer_cast(data.dets().data());
        detsum = (size_t*)thrust::raw_pointer_cast(data.detsums().data());
    }

    void set_mpi_size(size_t h_rank, size_t h_size)
    {
        assert(h_rank == 0 && h_size == 1);
    }
};


template <typename ElemT>
class MultSingleAlphaKernel : public MultKernelBase<ElemT>
{
public:
    using MultKernelBase<ElemT>::SubwarpSize;
protected:
	DetIndexMapThrust idxmap;
	DetIndexMapThrust tidxmap;
	ExcitationLookupThrust exidx;
public:
	MultSingleAlphaKernel (const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data,
				const DetIndexMapThrust& idxmap_in,
				const DetIndexMapThrust& tidxmap_in,
				const ExcitationLookupThrust& exidx_in)
		: MultKernelBase<ElemT>(v_wb, v_t, data), idxmap(idxmap_in), tidxmap(tidxmap_in), exidx(exidx_in) {}

    // kernel entry point
    __device__ __host__ void operator()(size_t arg_i)
    {
		constexpr int SUBWARP = SubwarpSize;
		uint32_t i    = static_cast<uint32_t>(arg_i / SUBWARP);
		int    lane = static_cast<int>(arg_i % SUBWARP);

		uint32_t ibst = idxmap.AdetToBdetSM[i];
		uint32_t idet = idxmap.AdetToDetSM[i];
		uint32_t ia   = idxmap.AdetIndex[i];
		uint32_t iast = ia;

		using WarpReduceSum = cub::WarpReduce<ElemT, SUBWARP>;
		typename WarpReduceSum::TempStorage temp_sum;
		ElemT thread_sum = ElemT(0);

		const uint32_t self_lo = exidx.SelfFromBdetOffset[ibst];
		const uint32_t self_hi = exidx.SelfFromBdetOffset[ibst + 1];
		const uint32_t ja_lo   = exidx.SinglesFromAdetOffset[ia];
		const uint32_t ja_hi   = exidx.SinglesFromAdetOffset[ia + 1];
		if (self_lo != self_hi && ja_lo != ja_hi) {
			uint32_t jbst = exidx.SelfFromBdetSM[self_lo];
			const uint32_t bdet_lo = tidxmap.BdetToDetOffset[jbst];
			const uint32_t bdet_hi = tidxmap.BdetToDetOffset[jbst + 1];
			// single alpha excitations — strided across SUBWARP lanes
			for (uint32_t ja = ja_lo + lane; ja < ja_hi; ja += SUBWARP) {
				uint32_t jast = exidx.SinglesFromAdetSM[ja];
				auto [found_a, idxa] = tidxmap.bdet_lower_bound(bdet_lo, bdet_hi, jast);
				if (found_a) {
					uint32_t jdet = tidxmap.BdetToDetSM[idxa];
					ElemT eij = this->OneExcite(this->det + idet * this->D_size,
											exidx.SinglesAdetCrAnSM[ja],
											exidx.SinglesAdetCrAnSM[ja + exidx.size_single_adet]);
					thread_sum += eij * this->twk[jdet];
				}
			}
		} // if there is same beta string

		ElemT total = WarpReduceSum(temp_sum).Sum(thread_sum);
		if (lane == 0 && total != ElemT(0)) this->wb[idet] += total;
	}
};

template <typename ElemT>
class MultDoubleAlphaKernel : public MultKernelBase<ElemT>
{
public:
    using MultKernelBase<ElemT>::SubwarpSize;
protected:
	DetIndexMapThrust idxmap;
	DetIndexMapThrust tidxmap;
	ExcitationLookupThrust exidx;
public:
	MultDoubleAlphaKernel (const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data,
				const DetIndexMapThrust& idxmap_in,
				const DetIndexMapThrust& tidxmap_in,
				const ExcitationLookupThrust& exidx_in)
		: MultKernelBase<ElemT>(v_wb, v_t, data), idxmap(idxmap_in), tidxmap(tidxmap_in), exidx(exidx_in) {}

    // kernel entry point
    __device__ __host__ void operator()(size_t arg_i)
    {
		constexpr int SUBWARP = SubwarpSize;
		uint32_t i    = static_cast<uint32_t>(arg_i / SUBWARP);
		int    lane = static_cast<int>(arg_i % SUBWARP);

		uint32_t ibst = idxmap.AdetToBdetSM[i];
		uint32_t idet = idxmap.AdetToDetSM[i];
		uint32_t ia   = idxmap.AdetIndex[i];
		uint32_t iast = ia;

		using WarpReduceSum = cub::WarpReduce<ElemT, SUBWARP>;
		typename WarpReduceSum::TempStorage temp_sum;
		ElemT thread_sum = ElemT(0);

		const uint32_t self_lo = exidx.SelfFromBdetOffset[ibst];
		const uint32_t self_hi = exidx.SelfFromBdetOffset[ibst + 1];
		const uint32_t ja_lo   = exidx.DoublesFromAdetOffset[ia];
		const uint32_t ja_hi   = exidx.DoublesFromAdetOffset[ia + 1];
		if (self_lo != self_hi && ja_lo != ja_hi) {
			uint32_t jbst = exidx.SelfFromBdetSM[self_lo];
			const uint32_t bdet_lo = tidxmap.BdetToDetOffset[jbst];
			const uint32_t bdet_hi = tidxmap.BdetToDetOffset[jbst + 1];
			// double alpha excitations — strided across SUBWARP lanes
			for (uint32_t ja = ja_lo + lane; ja < ja_hi; ja += SUBWARP) {
				uint32_t jast = exidx.DoublesFromAdetSM[ja];
				auto [found_a, idxa] = tidxmap.bdet_lower_bound(bdet_lo, bdet_hi, jast);
				if (found_a) {
					uint32_t jdet = tidxmap.BdetToDetSM[idxa];
					ElemT eij = this->TwoExciteFast(this->detsum + idet * this->D_size,
											exidx.DoublesAdetCrAnSM[ja],
											exidx.DoublesAdetCrAnSM[ja + exidx.size_double_adet],
											exidx.DoublesAdetCrAnSM[ja + exidx.size_double_adet * 2],
											exidx.DoublesAdetCrAnSM[ja + exidx.size_double_adet * 3]);
					thread_sum += eij * this->twk[jdet];
				}
			}
		} // if there is same beta string

		ElemT total = WarpReduceSum(temp_sum).Sum(thread_sum);
		if (lane == 0 && total != ElemT(0)) this->wb[idet] += total;
	}
};




template <typename ElemT>
class MultSingleBetaKernel : public MultKernelBase<ElemT>
{
public:
    using MultKernelBase<ElemT>::SubwarpSize;
protected:
	DetIndexMapThrust idxmap;
	DetIndexMapThrust tidxmap;
	ExcitationLookupThrust exidx;
public:
	MultSingleBetaKernel (const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data,
				const DetIndexMapThrust& idxmap_in,
				const DetIndexMapThrust& tidxmap_in,
				const ExcitationLookupThrust& exidx_in)
		: MultKernelBase<ElemT>(v_wb, v_t, data), idxmap(idxmap_in), tidxmap(tidxmap_in), exidx(exidx_in) {}

    // kernel entry point
    __device__ __host__ void operator()(size_t arg_i)
    {
		constexpr int SUBWARP = SubwarpSize;
		uint32_t i    = static_cast<uint32_t>(arg_i / SUBWARP);
		int    lane = static_cast<int>(arg_i % SUBWARP);

		uint32_t iast = idxmap.BdetToAdetSM[i];
		uint32_t idet = idxmap.BdetToDetSM[i];
		uint32_t ib   = idxmap.BdetIndex[i];
		uint32_t ibst = ib;

		using WarpReduceSum = cub::WarpReduce<ElemT, SUBWARP>;
		typename WarpReduceSum::TempStorage temp_sum;
		ElemT thread_sum = ElemT(0);

		const uint32_t self_lo = exidx.SelfFromAdetOffset[iast];
		const uint32_t self_hi = exidx.SelfFromAdetOffset[iast + 1];
		const uint32_t jb_lo   = exidx.SinglesFromBdetOffset[ib];
		const uint32_t jb_hi   = exidx.SinglesFromBdetOffset[ib + 1];
		if (self_lo != self_hi && jb_lo != jb_hi) {
			uint32_t jast = exidx.SelfFromAdetSM[self_lo];
			const uint32_t adet_lo = tidxmap.AdetToDetOffset[jast];
			const uint32_t adet_hi = tidxmap.AdetToDetOffset[jast + 1];
			// single beta excitations — strided across SUBWARP lanes
			for (uint32_t jb = jb_lo + lane; jb < jb_hi; jb += SUBWARP) {
				uint32_t jbst = exidx.SinglesFromBdetSM[jb];
				auto [found_b, idxb] = tidxmap.adet_lower_bound(adet_lo, adet_hi, jbst);
				if (found_b) {
					uint32_t jdet = tidxmap.AdetToDetSM[idxb];
					ElemT eij = this->OneExcite(this->det + idet * this->D_size,
											exidx.SinglesBdetCrAnSM[jb],
											exidx.SinglesBdetCrAnSM[jb + exidx.size_single_bdet]);
					thread_sum += eij * this->twk[jdet];
				}
			}
		} // if there is same beta string

		ElemT total = WarpReduceSum(temp_sum).Sum(thread_sum);
		if (lane == 0 && total != ElemT(0)) this->wb[idet] += total;
	}
};

template <typename ElemT>
class MultDoubleBetaKernel : public MultKernelBase<ElemT>
{
public:
    using MultKernelBase<ElemT>::SubwarpSize;
protected:
	DetIndexMapThrust idxmap;
	DetIndexMapThrust tidxmap;
	ExcitationLookupThrust exidx;
public:
	MultDoubleBetaKernel (const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data,
				const DetIndexMapThrust& idxmap_in,
				const DetIndexMapThrust& tidxmap_in,
				const ExcitationLookupThrust& exidx_in)
		: MultKernelBase<ElemT>(v_wb, v_t, data), idxmap(idxmap_in), tidxmap(tidxmap_in), exidx(exidx_in) {}

    // kernel entry point
    __device__ __host__ void operator()(size_t arg_i)
    {
		constexpr int SUBWARP = SubwarpSize;
		uint32_t i    = static_cast<uint32_t>(arg_i / SUBWARP);
		int    lane = static_cast<int>(arg_i % SUBWARP);

		uint32_t iast = idxmap.BdetToAdetSM[i];
		uint32_t idet = idxmap.BdetToDetSM[i];
		uint32_t ib   = idxmap.BdetIndex[i];
		uint32_t ibst = ib;

		using WarpReduceSum = cub::WarpReduce<ElemT, SUBWARP>;
		typename WarpReduceSum::TempStorage temp_sum;
		ElemT thread_sum = ElemT(0);

		const uint32_t self_lo = exidx.SelfFromAdetOffset[iast];
		const uint32_t self_hi = exidx.SelfFromAdetOffset[iast + 1];
		const uint32_t jb_lo   = exidx.DoublesFromBdetOffset[ib];
		const uint32_t jb_hi   = exidx.DoublesFromBdetOffset[ib + 1];
		if (self_lo != self_hi && jb_lo != jb_hi) {
			uint32_t jast = exidx.SelfFromAdetSM[self_lo];
			const uint32_t adet_lo = tidxmap.AdetToDetOffset[jast];
			const uint32_t adet_hi = tidxmap.AdetToDetOffset[jast + 1];
			// double beta excitations — strided across SUBWARP lanes
			for (uint32_t jb = jb_lo + lane; jb < jb_hi; jb += SUBWARP) {
				uint32_t jbst = exidx.DoublesFromBdetSM[jb];
				auto [found_b, idxb] = tidxmap.adet_lower_bound(adet_lo, adet_hi, jbst);
				if (found_b) {
					uint32_t jdet = tidxmap.AdetToDetSM[idxb];
					ElemT eij = this->TwoExciteFast(this->detsum + idet * this->D_size,
											exidx.DoublesBdetCrAnSM[jb],
											exidx.DoublesBdetCrAnSM[jb + exidx.size_double_bdet],
											exidx.DoublesBdetCrAnSM[jb + exidx.size_double_bdet * 2],
											exidx.DoublesBdetCrAnSM[jb + exidx.size_double_bdet * 3]);
					thread_sum += eij * this->twk[jdet];
				}
			}
		} // if there is same beta string

		ElemT total = WarpReduceSum(temp_sum).Sum(thread_sum);
		if (lane == 0 && total != ElemT(0)) this->wb[idet] += total;
	}
};


template <typename ElemT>
class MultAlphaBetaKernel : public MultKernelBase<ElemT>
{
public:
    static constexpr int SubwarpSize = SBD_GDB_AB_SUBWARP_SIZE;
    using MultKernelBase<ElemT>::BlockSize;
protected:
	DetIndexMapThrust idxmap;
	DetIndexMapThrust tidxmap;
	ExcitationLookupThrust exidx;
public:
	MultAlphaBetaKernel (const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data,
				const DetIndexMapThrust& idxmap_in,
				const DetIndexMapThrust& tidxmap_in,
				const ExcitationLookupThrust& exidx_in)
		: MultKernelBase<ElemT>(v_wb, v_t, data), idxmap(idxmap_in), tidxmap(tidxmap_in), exidx(exidx_in) {}

    // kernel entry point
    __device__ __host__ void operator()(size_t arg_i)
    {
		constexpr int SUBWARP   = SubwarpSize;
		constexpr int BLOCK     = BlockSize;
		constexpr int GROUPS    = BLOCK / SUBWARP;        // SUBWARP groups per block
		constexpr int BUF_PG    = 2 * SUBWARP;            // slots per group
		constexpr int BUF_TOTAL = 2 * BLOCK;              // = GROUPS * BUF_PG

		uint32_t i     = static_cast<uint32_t>(arg_i / SUBWARP); // SUBWARP=1 → i = arg_i
		int    lane  = static_cast<int>(arg_i % SUBWARP); // 0..SUBWARP-1 within group
		int    group = static_cast<int>(i % GROUPS);      // 0..GROUPS-1 within block
		int    buf_base = group * BUF_PG;

		uint32_t ibst = idxmap.AdetToBdetSM[i];
		uint32_t idet = idxmap.AdetToDetSM[i];
		uint32_t ia   = idxmap.AdetIndex[i];
		uint32_t iast = ia;

		using WarpReduceSum = cub::WarpReduce<ElemT, SUBWARP>;
		typename WarpReduceSum::TempStorage temp_sum;   // local; shuffle-only at p2 sizes
		ElemT thread_sum = ElemT(0);

		// SUBWARP-parallel (idxb, ja, k) accumulation buffer. Per-block
		// shared, sliced by SUBWARP group. Each group's SUBWARP lanes ballot
		// for valid candidates per probe round, append at distinct ranks
		// (no atomicAdd), then dispatch TwoExcite once the group's slice
		// holds ≥ SUBWARP entries.
		__shared__ uint32_t s_idxb [BUF_TOTAL];
		__shared__ uint32_t s_ja   [BUF_TOTAL];
		__shared__ uint32_t s_k    [BUF_TOTAL];
		// s_count lives in a register: every lane computes __popc(warp_ballot)
		// independently, so all lanes' copies stay identical without any
		// shared-memory round-trip or __syncwarp for the count itself.
		int s_count = 0;

		// Per-SUBWARP ballot mask — covers this group's lanes within the
		// containing CUDA warp. With multiple groups per warp (SUBWARP < 32),
		// each group masks the warp-wide ballot to its own bit-range.
		const unsigned lane_in_warp  = static_cast<unsigned>(threadIdx.x) % 32u;
		const unsigned group_in_warp = lane_in_warp / static_cast<unsigned>(SUBWARP);
		const unsigned subwarp_lanes = (SUBWARP == 32) ? 0xFFFFFFFFu
		                                               : ((1u << SUBWARP) - 1u);
		const unsigned subwarp_mask  = subwarp_lanes << (group_in_warp * SUBWARP);

		// k_lo/k_hi/k_iters depend only on ibst (constant for this thread);
		// hoist outside the ja loop so the compiler sees they are loop-invariant
		// even though SinglesFromBdetOffset is a raw device pointer.
		const uint32_t k_lo    = exidx.SinglesFromBdetOffset[ibst];
		const uint32_t k_hi    = exidx.SinglesFromBdetOffset[ibst + 1];
		const uint32_t k_iters = (k_hi - k_lo + SUBWARP - 1) / SUBWARP;
		const uint32_t ja_lo   = exidx.SinglesFromAdetOffset[ia];
		const uint32_t ja_hi   = exidx.SinglesFromAdetOffset[ia + 1];

		// Outer ja loop. Inner k loop runs k_iters times *uniformly* across
		// all SUBWARP lanes of the group (k_lane = k_lo + it*SUBWARP + lane,
		// guarded by `in_range = k_lane < k_hi`). This keeps __ballot_sync
		// reachable by every lane on every iteration — fixing the
		// divergent-loop-exit bug in v1.
		// adet_lower_bound returns {found, ie}; ie < AdetToDetOffset[jast+1] whenever found.
		// (idxb - AdetToDetOffset[jast]) is in-range whenever valid.
		if (k_lo != k_hi) for (uint32_t ja = ja_lo; ja < ja_hi; ja++) {
			uint32_t jast    = exidx.SinglesFromAdetSM[ja];
			const uint32_t adet_lo = tidxmap.AdetToDetOffset[jast];
			const uint32_t adet_hi = tidxmap.AdetToDetOffset[jast + 1];

			for (uint32_t it = 0; it < k_iters; it++) {
				uint32_t k             = k_lo + it * SUBWARP + lane;
				uint32_t jbst          = exidx.SinglesFromBdetSM[std::min(k, k_hi - 1)];
				auto [found, idxb]     = tidxmap.adet_lower_bound(adet_lo, adet_hi, jbst);
				bool     valid         = k < k_hi && found;

				// Per-group ballot. __ballot_sync(subwarp_mask,...) zeroes
				// bits outside the mask so warp_ballot is already confined
				// to this group; no shift needed for rank or total.
				unsigned warp_ballot = __ballot_sync(subwarp_mask, valid);

				if (valid) {
					int rank = __popc(warp_ballot & ((1u << lane_in_warp) - 1u));
					int slot = buf_base + s_count + rank;
					s_idxb[slot] = idxb;
					s_ja  [slot] = ja;
					s_k   [slot] = k;
				}
				// All lanes compute __popc(warp_ballot) identically — no syncwarp.
				s_count += __popc(warp_ballot);

				// Dispatch when a full SUBWARP batch has accumulated.
				// s_count < 2*SUBWARP at this point (invariant: s_count was
				// < SUBWARP before the ballot, at most SUBWARP added), so
				// if is sufficient — at most one dispatch per k-iteration.
				if (s_count >= SUBWARP) {
					// Fence: ensure accumulate writes above are visible to all
					// lanes before dispatch reads below.
					__syncwarp(subwarp_mask);
					if (lane < s_count) {
						int      my_slot = buf_base + lane;
						uint32_t my_idxb = s_idxb[my_slot];
						uint32_t my_ja   = s_ja  [my_slot];
						uint32_t my_k    = s_k   [my_slot];
						uint32_t jdet    = tidxmap.AdetToDetSM[my_idxb];
						ElemT eij = this->TwoExciteFast(this->detsum + idet * this->D_size,
						                    exidx.SinglesAdetCrAnSM[my_ja],
						                    exidx.SinglesBdetCrAnSM[my_k],
						                    exidx.SinglesAdetCrAnSM[my_ja + exidx.size_single_adet],
						                    exidx.SinglesBdetCrAnSM[my_k + exidx.size_single_bdet]);
						thread_sum += eij * this->twk[jdet];
					}
					// No __syncwarp between dispatch and compact: each lane reads/writes
					// only its own slot (dispatch reads buf_base+lane; compact reads
					// buf_base+SUBWARP+lane and writes buf_base+lane — non-overlapping
					// per-lane ranges, so no cross-lane hazard here). The next
					// __ballot_sync (next k-iteration) or post-loop __syncwarp fences
					// these compact writes before subsequent reads.
					s_idxb[buf_base + lane] = s_idxb[buf_base + SUBWARP + lane];
					s_ja  [buf_base + lane] = s_ja  [buf_base + SUBWARP + lane];
					s_k   [buf_base + lane] = s_k   [buf_base + SUBWARP + lane];
					s_count -= SUBWARP;
				}
			}
		}

		// Drain any remaining candidates left in the buffer after all ja/k loops.
		if (s_count > 0) {
			__syncwarp(subwarp_mask);
			if (lane < s_count) {
				int      my_slot = buf_base + lane;
				uint32_t my_idxb = s_idxb[my_slot];
				uint32_t my_ja   = s_ja  [my_slot];
				uint32_t my_k    = s_k   [my_slot];
				uint32_t jdet    = tidxmap.AdetToDetSM[my_idxb];
				ElemT eij = this->TwoExciteFast(this->detsum + idet * this->D_size,
				                    exidx.SinglesAdetCrAnSM[my_ja],
				                    exidx.SinglesBdetCrAnSM[my_k],
				                    exidx.SinglesAdetCrAnSM[my_ja + exidx.size_single_adet],
				                    exidx.SinglesBdetCrAnSM[my_k + exidx.size_single_bdet]);
				thread_sum += eij * this->twk[jdet];
			}
		}

		ElemT total = WarpReduceSum(temp_sum).Sum(thread_sum);
		if (lane == 0 && total != ElemT(0)) this->wb[idet] += total;
	}
};


template <typename ElemT>
class Wb_init_kernel {
protected:
    ElemT* Wb;
    ElemT* hii;
    ElemT* T;
public:
    Wb_init_kernel(thrust::device_vector<ElemT>& Wb_in, const thrust::device_vector<ElemT>& hii_in, const thrust::device_vector<ElemT>& T_in)
    {
        Wb = (ElemT*)thrust::raw_pointer_cast(Wb_in.data());
        hii = (ElemT*)thrust::raw_pointer_cast(hii_in.data());
        T = (ElemT*)thrust::raw_pointer_cast(T_in.data());
    }
    __host__ __device__ void operator()(size_t i)
    {
        Wb[i] += hii[i] * T[i];
    }
};

template <typename ElemT>
void MultGDBThrust<ElemT>::run(	const thrust::device_vector<ElemT> &hii,
								const thrust::device_vector<ElemT> &wk,
								thrust::device_vector<ElemT> &wb)
{
	int mpi_size_h;
	MPI_Comm_size(this->h_comm(), &mpi_size_h);
	int mpi_rank_h;
	MPI_Comm_rank(this->h_comm(), &mpi_rank_h);
	int mpi_size_b;
	MPI_Comm_size(this->b_comm(), &mpi_size_b);
	int mpi_rank_b;
	MPI_Comm_rank(this->b_comm(), &mpi_rank_b);
	int mpi_size_t;
	MPI_Comm_size(this->t_comm(), &mpi_size_t);
	int mpi_rank_t;
	MPI_Comm_rank(this->t_comm(), &mpi_rank_t);

	// double buffering for MPI slide
	thrust::device_vector<ElemT> twk[2];
	thrust::device_vector<uint32_t> tidxmap_storage[2];
	DetIndexMapThrust tidxmap[2];
	int active_buf = 0;
    int recv_buf = 1;

	if (exidx[0].slide != 0) {
		sbd::gdb::MpiSlide(idxmap, idxmap_storage, tidxmap[active_buf], tidxmap_storage[active_buf], -exidx[0].slide, this->b_comm());
		sbd::MpiSlide(wk, twk[active_buf], -exidx[0].slide, this->b_comm());
	} else {
		tidxmap[active_buf].copy(tidxmap_storage[active_buf], idxmap, idxmap_storage);
		twk[active_buf] = wk;
	}

	// Pre-allocate both double buffers to the global maximum ket/map size before
	// the task loop.  This eliminates per-task thrust::device_vector::resize()
	// calls inside ExchangeAsync, which previously launched a Thrust fill-to-zero
	// kernel followed by cudaDeviceSynchronize() — serializing GPU/MPI overlap on
	// every task where the recv buffer needed to grow.
	//
	// ORDERING: pre-allocation MUST come before Wb_init_kernel.  Wb_init_kernel
	// captures twk[active_buf].data() at construction time; if resize() ran after
	// the kernel was launched it could reallocate the buffer and free the old device
	// memory while the kernel is still running on it.
	//
	// Because send.size() / send_map_storage.size() would then equal global_max
	// rather than the actual ket/map element count, the caller-supplied logical
	// sizes (twk_ket_size[] / twk_map_size[]) are passed explicitly to
	// ExchangeAsync and updated after each successful Sync().
	size_t twk_ket_size[2];
	size_t twk_map_size[2];
	{
		size_t local_ket_size = twk[active_buf].size();
		size_t global_max_ket_size;
		MPI_Allreduce(&local_ket_size, &global_max_ket_size, 1, SBD_MPI_SIZE_T, MPI_MAX, this->b_comm());

		size_t local_map_size = tidxmap_storage[active_buf].size();
		size_t global_max_map_size;
		MPI_Allreduce(&local_map_size, &global_max_map_size, 1, SBD_MPI_SIZE_T, MPI_MAX, this->b_comm());

		// Logical sizes tracked independently of vector .size() (which = global_max).
		twk_ket_size[active_buf] = local_ket_size;
		twk_ket_size[recv_buf]   = global_max_ket_size;  // placeholder; set after first recv
		twk_map_size[active_buf] = local_map_size;
		twk_map_size[recv_buf]   = global_max_map_size;

		// Resize both ket buffers to global max.
		twk[active_buf].resize(global_max_ket_size);
		twk[recv_buf].resize(global_max_ket_size);

		// Resize active map storage.  If resize() triggers reallocation the data
		// moves to a new device address, which would invalidate the raw pointers
		// already stored in tidxmap[active_buf].  Detect and fix up.
		{
			uint32_t* old_base = thrust::raw_pointer_cast(tidxmap_storage[active_buf].data());
			tidxmap_storage[active_buf].resize(global_max_map_size);
			uint32_t* new_base = thrust::raw_pointer_cast(tidxmap_storage[active_buf].data());
			if (new_base != old_base) {
				ptrdiff_t delta = new_base - old_base;
				tidxmap[active_buf].AdetToDetOffset += delta;
				tidxmap[active_buf].BdetToDetOffset += delta;
				tidxmap[active_buf].AdetIndex       += delta;
				tidxmap[active_buf].BdetIndex       += delta;
				tidxmap[active_buf].AdetToBdetSM    += delta;
				tidxmap[active_buf].AdetToDetSM     += delta;
				tidxmap[active_buf].BdetToAdetSM    += delta;
				tidxmap[active_buf].BdetToDetSM     += delta;
			}
		}
		tidxmap_storage[recv_buf].resize(global_max_map_size);
		// Ensure all fill kernels complete before MPI_Irecv writes into these
		// buffers via GPUDirect RDMA.
		cudaDeviceSynchronize();
	}

	// Wb_init_kernel AFTER pre-allocation so it captures the (potentially
	// reallocated) twk[active_buf] pointer.  Use twk_ket_size[active_buf]
	// — not twk[active_buf].size() which equals global_max after resize.
	if (mpi_rank_t == 0) {
        auto ci = thrust::counting_iterator<size_t>(0);
        thrust::for_each_n(thrust::device, ci, twk_ket_size[active_buf], Wb_init_kernel(wb, hii, twk[active_buf]));
	}

	// Non-blocking stream keeps mult kernels off the default stream so
	// Cray MPICH GPU-direct DMA (which syncs the default stream) does not
	// stall waiting for kernel completion.
	cudaStream_t compute_stream;
	cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);

	MpiSlider<ElemT> slider;
	using _ck = std::chrono::steady_clock;
	for (size_t task = 0; task < exidx.size(); task++) {
		// H19 ordering (kernels first, then ExchangeAsync) with kernels on
		// compute_stream.  Cray MPICH syncs the default stream before DMA;
		// since kernels run on compute_stream the default stream is always
		// clean when ExchangeAsync posts MPI_Isend.
		double _t_launch, _t_exch, _t_sync, _t_gpu;

		{ auto _t0 = _ck::now();

		// single alpha excitations
		MultSingleAlphaKernel kernel_single_alpha(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_single_alpha.set_mpi_size(mpi_rank_h, mpi_size_h);
		launch_mult_for_each_n(idxmap.size_adet, kernel_single_alpha, compute_stream);

		// double alpha excitations
		MultDoubleAlphaKernel kernel_double_alpha(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_double_alpha.set_mpi_size(mpi_rank_h, mpi_size_h);
		launch_mult_for_each_n(idxmap.size_adet, kernel_double_alpha, compute_stream);

		// single beta excitations
		MultSingleBetaKernel kernel_single_beta(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_single_beta.set_mpi_size(mpi_rank_h, mpi_size_h);
		launch_mult_for_each_n(idxmap.size_bdet, kernel_single_beta, compute_stream);

		// double beta excitations
		MultDoubleBetaKernel kernel_double_beta(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_double_beta.set_mpi_size(mpi_rank_h, mpi_size_h);
		launch_mult_for_each_n(idxmap.size_bdet, kernel_double_beta, compute_stream);

		// alpha-beta excitations
		MultAlphaBetaKernel kernel_alpha_beta(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_alpha_beta.set_mpi_size(mpi_rank_h, mpi_size_h);
		launch_mult_for_each_n(idxmap.size_adet, kernel_alpha_beta, compute_stream);

		_t_launch = std::chrono::duration<double,std::milli>(_ck::now()-_t0).count(); }

		if (task < exidx.size() - 1) {
			int slide = exidx[task].slide - exidx[task + 1].slide;
			{ auto _t0 = _ck::now();
			  slider.ExchangeAsync(
			      twk[active_buf], twk_ket_size[active_buf],
			      twk[recv_buf],
			      tidxmap[active_buf], tidxmap_storage[active_buf], twk_map_size[active_buf],
			      tidxmap[recv_buf], tidxmap_storage[recv_buf],
			      slide, this->b_comm(), (int)task);
			  _t_exch = std::chrono::duration<double,std::milli>(_ck::now()-_t0).count(); }
		} else {
			_t_exch = 0.0;
		}

		if (task < exidx.size() - 1) {
			{ auto _t0 = _ck::now();
			  if (slider.Sync()) {
			    twk_ket_size[recv_buf] = slider.get_recv_size();
			    twk_map_size[recv_buf] = slider.get_recv_size_map();
			    int t = active_buf; active_buf = recv_buf; recv_buf = t;
			  }
			  _t_sync = std::chrono::duration<double,std::milli>(_ck::now()-_t0).count(); }
		} else {
			_t_sync = 0.0;
		}

		{ auto _t0 = _ck::now();
		  cudaStreamSynchronize(compute_stream);
		  _t_gpu = std::chrono::duration<double,std::milli>(_ck::now()-_t0).count(); }

		fprintf(stderr, "mult_task_ms: rank_b=%d task=%zu"
		        " launch=%.1f exch=%.1f sync=%.1f gpu_sync=%.1f\n",
		        mpi_rank_b, task, _t_launch, _t_exch, _t_sync, _t_gpu);
	} // end task for loop

	cudaStreamDestroy(compute_stream);

	if (mpi_size_t > 1)
		MpiAllreduce(wb, MPI_SUM, this->t_comm());
	if (mpi_size_h > 1)
		MpiAllreduce(wb, MPI_SUM, this->h_comm());
}



template <typename ElemT>
void mult(const std::vector<ElemT> &hii,
			const std::vector<ElemT> &wk,
			std::vector<ElemT> &wb,
			size_t bit_length,
			size_t norb,
			const std::vector<std::vector<size_t>> &det,
			const DetIndexMap &idxmap,
			const std::vector<ExcitationLookup> &exidx,
			const ElemT &I0,
			const oneInt<ElemT> &I1,
			const twoInt<ElemT> &I2,
			MPI_Comm h_comm,
			MPI_Comm b_comm,
			MPI_Comm t_comm)
{
	////  temporal implementation until Davidson for GPU is not ready
	MultGDBThrust<ElemT> data;
	data.Init(bit_length, norb, det, idxmap, exidx, I0, I1, I2, h_comm, b_comm, t_comm);

	thrust::device_vector<ElemT> wb_dev(wb.size());
	thrust::device_vector<ElemT> wk_dev(wk.size());
	thrust::device_vector<ElemT> hii_dev(hii.size());

	thrust::copy_n(wb.begin(), wb.size(), wb_dev.begin());
	thrust::copy_n(wk.begin(), wk.size(), wk_dev.begin());
	thrust::copy_n(hii.begin(), hii.size(), hii_dev.begin());

	data.run(hii_dev, wk_dev, wb_dev);

	thrust::copy_n(wb_dev.begin(), wb.size(), wb.begin());
	////

} // end function for mult

template <typename ElemT>
class MakeQChamDiagTermKernel : public MultKernelBase<ElemT>
{
protected:
    ElemT* hii;
public:
    MakeQChamDiagTermKernel(thrust::device_vector<ElemT>& hii_in, const MultGDBThrust<ElemT>& data)
                        : MultKernelBase<ElemT>(data)
    {
        hii = (ElemT*)thrust::raw_pointer_cast(hii_in.data());
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t* Det = this->det + i * this->D_size;
        hii[i] = this->ZeroExcite(Det);
    }
};


template <typename ElemT>
void MultGDBThrust<ElemT>::makeQChamDiagTerms(thrust::device_vector<ElemT> &hii)
{
    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(this->h_comm_, &mpi_rank_h);
    MPI_Comm_size(this->h_comm_, &mpi_size_h);

    // dets_ packs n_dets bit-strings of length D_size_ each; iterate over
    // n_dets, not dets_.size(), or the kernel will overrun dets_.
    const size_t n_dets = dets_.size() / this->D_size_;
    hii.resize(n_dets, ElemT(0.0));

    MakeQChamDiagTermKernel kernel(hii, *this);
	kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
    auto ci = thrust::counting_iterator<size_t>(0);
    thrust::for_each_n(thrust::device, ci, n_dets, kernel);
}


template <typename ElemT>
void mult(const std::vector<ElemT> & hii,
		const std::vector<std::vector<size_t*>> & ih,
		const std::vector<std::vector<size_t*>> & jh,
		const std::vector<std::vector<ElemT*>> & hij,
		const std::vector<std::vector<size_t>> & len,
		const std::vector<int> & slide,
		const std::vector<ElemT> & wk,
		std::vector<ElemT> & wb,
		MPI_Comm h_comm,
		MPI_Comm b_comm,
		MPI_Comm t_comm)
{
	//this is not used for Thrust
}


} // end namespace gdb

} // end namespace sbd

#endif
