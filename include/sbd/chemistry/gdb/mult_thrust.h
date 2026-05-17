/**
@file sbd/chemistry/tpb/mult.h
@brief Function to perform Hamiltonian operation for general determinant basis
*/
#ifndef SBD_CHEMISTRY_GDB_MULT_THRUST_H
#define SBD_CHEMISTRY_GDB_MULT_THRUST_H


#include "sbd/framework/mpi_utility_thrust.h"

// SUBWARP threading for MultAlphaBetaKernel (track gh-A).
// SBD_GDB_SUBWARP_SIZE selects the threading granularity:
//   1            → stride=1, behaves like the original code path
//                  (cub::WarpReduce<T,1> is identity; no cross-lane comm)
//   2 / 4 / 8 / 16 / 32 → strided inner-k loop + cub::WarpReduce reduction
//                  + per-iteration ballot-based candidate buffer.
// Single code path for all sizes — no #ifdef'd duplicate body.
#ifndef SBD_GDB_SUBWARP_SIZE
  #define SBD_GDB_SUBWARP_SIZE 1
#endif
static_assert(SBD_GDB_SUBWARP_SIZE == 1  ||
              SBD_GDB_SUBWARP_SIZE == 2  ||
              SBD_GDB_SUBWARP_SIZE == 4  ||
              SBD_GDB_SUBWARP_SIZE == 8  ||
              SBD_GDB_SUBWARP_SIZE == 16 ||
              SBD_GDB_SUBWARP_SIZE == 32,
              "SBD_GDB_SUBWARP_SIZE must be 1, 2, 4, 8, 16, or 32");
#include <cub/warp/warp_reduce.cuh>

namespace sbd
{
namespace gdb
{

#ifndef SBD_MULT_BLOCK_SIZE
  #define SBD_MULT_BLOCK_SIZE 32
#endif
static_assert(SBD_MULT_BLOCK_SIZE == 32  ||
              SBD_MULT_BLOCK_SIZE == 64  ||
              SBD_MULT_BLOCK_SIZE == 128 ||
              SBD_MULT_BLOCK_SIZE == 256,
              "SBD_MULT_BLOCK_SIZE must be 32, 64, 128, or 256");
static_assert(SBD_MULT_BLOCK_SIZE % SBD_GDB_SUBWARP_SIZE == 0,
              "SBD_MULT_BLOCK_SIZE must be a multiple of SBD_GDB_SUBWARP_SIZE.");

// SBD_MULT_MIN_BLOCKS_PER_SM tunes the second argument of
// __launch_bounds__ on mult_for_each_n_kernel — the
// minBlocksPerMultiprocessor hint the compiler uses to pick a register
// budget per thread. Default 32 = H100 hardware maximum; lower values
// trade occupancy for a larger per-thread register budget.
//   threads/SM = SBD_MULT_BLOCK_SIZE × SBD_MULT_MIN_BLOCKS_PER_SM
//   regs/thread budget ≈ 65536 / threads/SM
// Examples (BS=64):
//   mbpsm=32 → 2048 threads/SM (100% occupancy); 32 regs/thread
//   mbpsm=24 → 1536 threads/SM (75% occupancy); 42 regs/thread
//   mbpsm=20 → 1280 threads/SM (62.5% occupancy); 51 regs/thread
//   mbpsm=16 → 1024 threads/SM (50% occupancy); 64 regs/thread
#ifndef SBD_MULT_MIN_BLOCKS_PER_SM
  #define SBD_MULT_MIN_BLOCKS_PER_SM 32
#endif
static_assert(SBD_MULT_MIN_BLOCKS_PER_SM >= 1 &&
              SBD_MULT_MIN_BLOCKS_PER_SM <= 32,
              "SBD_MULT_MIN_BLOCKS_PER_SM must be in [1, 32] (H100 hw max).");

// Custom blocksize-tunable launcher for the GDB Mult kernels. Replaces
// thrust::for_each_n at the run()-level call sites with a fixed-block
// launch so the v2 SUBWARP-parallel kernel's shared-memory sizing
// (`BUF_TOTAL = 2 * SBD_MULT_BLOCK_SIZE`) matches the actual block size.
template <int BlockSize, typename Functor>
__global__ __launch_bounds__(BlockSize, SBD_MULT_MIN_BLOCKS_PER_SM)
void mult_for_each_n_kernel(size_t n, Functor functor)
{
    size_t i = static_cast<size_t>(blockIdx.x) * BlockSize + threadIdx.x;
    if (i < n) functor(i);
}

template <int BlockSize, typename Functor>
inline void launch_mult_for_each_n(size_t n, Functor functor)
{
    if (n == 0) return;
    const size_t grid = (n + BlockSize - 1) / BlockSize;
    mult_for_each_n_kernel<BlockSize><<<grid, BlockSize>>>(n, functor);
    // E1 (advice-from-parent §5): force per-launch stream serialization to
    // test the stream-ordering hypothesis. If this resolves the b_comm>=3
    // Davidson hang, the legacy default stream is the contention surface.
    cudaDeviceSynchronize();
}

template <typename ElemT>
class MultGDBThrust : public sbd::MultBase<ElemT> {
protected:
	thrust::device_vector<uint32_t> idxmap_storage;
	DetIndexMapThrust idxmap;
    thrust::device_vector<size_t> dets_;
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

	// copy indexmap
	idxmap = DetIndexMapThrust(idxmap_storage, idxmap_in);
}

template <typename ElemT>
class MultKernelBase : public DeterminantKernels<ElemT> {
protected:
    ElemT *wb;
    ElemT* twk;
    size_t mpi_rank_h;
    size_t mpi_size_h;
    size_t* det;
public:
    MultKernelBase() {}

    MultKernelBase( const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultGDBThrust<ElemT>& data
                ) : DeterminantKernels<ElemT>(data.bit_length(), data.norbs(), data.I0(), data.I1(), data.I2())
    {
        wb = (ElemT*)thrust::raw_pointer_cast(v_wb.data());
        twk = (ElemT*)thrust::raw_pointer_cast(v_t.data());
        det = (size_t*)thrust::raw_pointer_cast(data.dets().data());
    }

    MultKernelBase(const MultGDBThrust<ElemT>& data)
                 : DeterminantKernels<ElemT>(data.bit_length(), data.norbs(), data.I0(), data.I1(), data.I2())
    {
        det = (size_t*)thrust::raw_pointer_cast(data.dets().data());
    }

    void set_mpi_size(size_t h_rank, size_t h_size)
    {
        mpi_rank_h = h_rank;
        mpi_size_h = h_size;
    }
};


template <typename ElemT>
class MultSingleAlphaKernel : public MultKernelBase<ElemT>
{
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
		constexpr int SUBWARP = SBD_GDB_SUBWARP_SIZE;
		size_t i    = arg_i / SUBWARP;
		int    lane = static_cast<int>(arg_i % SUBWARP);

		uint32_t ibst = idxmap.AdetToBdetSM[i];
		uint32_t idet = idxmap.AdetToDetSM[i];
		uint32_t ia   = idxmap.AdetIndex[i];
		uint32_t iast = ia;
		if (idet % this->mpi_size_h != this->mpi_rank_h)
			return;

		using WarpReduceSum = cub::WarpReduce<ElemT, SUBWARP>;
		typename WarpReduceSum::TempStorage temp_sum;
		ElemT thread_sum = ElemT(0);

		if (exidx.SelfFromBdetOffset[ibst] != exidx.SelfFromBdetOffset[ibst + 1]) {
			uint32_t jbst = exidx.SelfFromBdetSM[exidx.SelfFromBdetOffset[ibst]];
			// single alpha excitations — strided across SUBWARP lanes
			for (uint32_t ja = exidx.SinglesFromAdetOffset[ia] + lane;
			              ja < exidx.SinglesFromAdetOffset[ia + 1];
			              ja += SUBWARP) {
				uint32_t jast = exidx.SinglesFromAdetSM[ja];
				int64_t idxa = tidxmap.bdet_lower_bound(jbst, jast);
				if (idxa >= 0) {
					if (jast != tidxmap.BdetToAdetSM[idxa])
						continue;
					uint32_t jdet = tidxmap.BdetToDetSM[idxa];
					ElemT eij = this->OneExcite(this->det + idet * this->D_size,
											exidx.SinglesAdetCrAnSM[ja],
											exidx.SinglesAdetCrAnSM[ja + exidx.size_single_adet]);
					thread_sum += eij * this->twk[jdet];
				}
			}
		} // if there is same beta string

		ElemT total = WarpReduceSum(temp_sum).Sum(thread_sum);
		if (lane == 0) this->wb[idet] += total;
	}
};

template <typename ElemT>
class MultDoubleAlphaKernel : public MultKernelBase<ElemT>
{
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
		constexpr int SUBWARP = SBD_GDB_SUBWARP_SIZE;
		size_t i    = arg_i / SUBWARP;
		int    lane = static_cast<int>(arg_i % SUBWARP);

		uint32_t ibst = idxmap.AdetToBdetSM[i];
		uint32_t idet = idxmap.AdetToDetSM[i];
		uint32_t ia   = idxmap.AdetIndex[i];
		uint32_t iast = ia;
		if (idet % this->mpi_size_h != this->mpi_rank_h)
			return;

		using WarpReduceSum = cub::WarpReduce<ElemT, SUBWARP>;
		typename WarpReduceSum::TempStorage temp_sum;
		ElemT thread_sum = ElemT(0);

		if (exidx.SelfFromBdetOffset[ibst] != exidx.SelfFromBdetOffset[ibst + 1]) {
			uint32_t jbst = exidx.SelfFromBdetSM[exidx.SelfFromBdetOffset[ibst]];
			// double alpha excitations — strided across SUBWARP lanes
			for (uint32_t ja = exidx.DoublesFromAdetOffset[ia] + lane;
			              ja < exidx.DoublesFromAdetOffset[ia + 1];
			              ja += SUBWARP) {
				uint32_t jast = exidx.DoublesFromAdetSM[ja];
				int64_t idxa = tidxmap.bdet_lower_bound(jbst, jast);
				if (idxa >= 0) {
					if (jast != tidxmap.BdetToAdetSM[idxa])
						continue;
					uint32_t jdet = tidxmap.BdetToDetSM[idxa];
					ElemT eij = this->TwoExcite(this->det + idet * this->D_size,
											exidx.DoublesAdetCrAnSM[ja],
											exidx.DoublesAdetCrAnSM[ja + exidx.size_double_adet],
											exidx.DoublesAdetCrAnSM[ja + exidx.size_double_adet * 2],
											exidx.DoublesAdetCrAnSM[ja + exidx.size_double_adet * 3]);
					thread_sum += eij * this->twk[jdet];
				}
			}
		} // if there is same beta string

		ElemT total = WarpReduceSum(temp_sum).Sum(thread_sum);
		if (lane == 0) this->wb[idet] += total;
	}
};




template <typename ElemT>
class MultSingleBetaKernel : public MultKernelBase<ElemT>
{
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
		constexpr int SUBWARP = SBD_GDB_SUBWARP_SIZE;
		size_t i    = arg_i / SUBWARP;
		int    lane = static_cast<int>(arg_i % SUBWARP);

		uint32_t iast = idxmap.BdetToAdetSM[i];
		uint32_t idet = idxmap.BdetToDetSM[i];
		uint32_t ib   = idxmap.BdetIndex[i];
		uint32_t ibst = ib;
		if (idet % this->mpi_size_h != this->mpi_rank_h)
			return;

		using WarpReduceSum = cub::WarpReduce<ElemT, SUBWARP>;
		typename WarpReduceSum::TempStorage temp_sum;
		ElemT thread_sum = ElemT(0);

		if (exidx.SelfFromAdetOffset[iast] != exidx.SelfFromAdetOffset[iast + 1]) {
			uint32_t jast = exidx.SelfFromAdetSM[exidx.SelfFromAdetOffset[iast]];
			// single beta excitations — strided across SUBWARP lanes
			for (uint32_t jb = exidx.SinglesFromBdetOffset[ib] + lane;
			              jb < exidx.SinglesFromBdetOffset[ib + 1];
			              jb += SUBWARP) {
				uint32_t jbst = exidx.SinglesFromBdetSM[jb];
				int64_t idxb = tidxmap.adet_lower_bound(jast, jbst);
				if (idxb >= 0) {
					if (jbst != tidxmap.AdetToBdetSM[idxb])
						continue;
					uint32_t jdet = tidxmap.AdetToDetSM[idxb];
					ElemT eij = this->OneExcite(this->det + idet * this->D_size,
											exidx.SinglesBdetCrAnSM[jb],
											exidx.SinglesBdetCrAnSM[jb + exidx.size_single_bdet]);
					thread_sum += eij * this->twk[jdet];
				}
			}
		} // if there is same beta string

		ElemT total = WarpReduceSum(temp_sum).Sum(thread_sum);
		if (lane == 0) this->wb[idet] += total;
	}
};

template <typename ElemT>
class MultDoubleBetaKernel : public MultKernelBase<ElemT>
{
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
		constexpr int SUBWARP = SBD_GDB_SUBWARP_SIZE;
		size_t i    = arg_i / SUBWARP;
		int    lane = static_cast<int>(arg_i % SUBWARP);

		uint32_t iast = idxmap.BdetToAdetSM[i];
		uint32_t idet = idxmap.BdetToDetSM[i];
		uint32_t ib   = idxmap.BdetIndex[i];
		uint32_t ibst = ib;
		if (idet % this->mpi_size_h != this->mpi_rank_h)
			return;

		using WarpReduceSum = cub::WarpReduce<ElemT, SUBWARP>;
		typename WarpReduceSum::TempStorage temp_sum;
		ElemT thread_sum = ElemT(0);

		if (exidx.SelfFromAdetOffset[iast] != exidx.SelfFromAdetOffset[iast + 1]) {
			uint32_t jast = exidx.SelfFromAdetSM[exidx.SelfFromAdetOffset[iast]];
			// double beta excitations — strided across SUBWARP lanes
			for (uint32_t jb = exidx.DoublesFromBdetOffset[ib] + lane;
			              jb < exidx.DoublesFromBdetOffset[ib + 1];
			              jb += SUBWARP) {
				uint32_t jbst = exidx.DoublesFromBdetSM[jb];
				int64_t idxb = tidxmap.adet_lower_bound(jast, jbst);
				if (idxb >= 0) {
					if (jbst != tidxmap.AdetToBdetSM[idxb])
						continue;
					uint32_t jdet = tidxmap.AdetToDetSM[idxb];
					ElemT eij = this->TwoExcite(this->det + idet * this->D_size,
											exidx.DoublesBdetCrAnSM[jb],
											exidx.DoublesBdetCrAnSM[jb + exidx.size_double_bdet],
											exidx.DoublesBdetCrAnSM[jb + exidx.size_double_bdet * 2],
											exidx.DoublesBdetCrAnSM[jb + exidx.size_double_bdet * 3]);
					thread_sum += eij * this->twk[jdet];
				}
			}
		} // if there is same beta string

		ElemT total = WarpReduceSum(temp_sum).Sum(thread_sum);
		if (lane == 0) this->wb[idet] += total;
	}
};


template <typename ElemT>
class MultAlphaBetaKernel : public MultKernelBase<ElemT>
{
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
		constexpr int SUBWARP   = SBD_GDB_SUBWARP_SIZE;
		constexpr int BLOCK     = SBD_MULT_BLOCK_SIZE;
		constexpr int GROUPS    = BLOCK / SUBWARP;        // SUBWARP groups per block
		constexpr int BUF_PG    = 2 * SUBWARP;            // slots per group
		constexpr int BUF_TOTAL = 2 * BLOCK;              // = GROUPS * BUF_PG

		size_t i     = arg_i / SUBWARP;                   // SUBWARP=1 → i = arg_i
		int    lane  = static_cast<int>(arg_i % SUBWARP); // 0..SUBWARP-1 within group
		int    group = static_cast<int>(i % GROUPS);      // 0..GROUPS-1 within block
		int    buf_base = group * BUF_PG;

		uint32_t ibst = idxmap.AdetToBdetSM[i];
		uint32_t idet = idxmap.AdetToDetSM[i];
		uint32_t ia   = idxmap.AdetIndex[i];
		uint32_t iast = ia;
		if (idet % this->mpi_size_h != this->mpi_rank_h)
			return;

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
		__shared__ int     s_count[GROUPS];
		if (lane == 0) s_count[group] = 0;

		// Per-SUBWARP ballot mask — covers this group's lanes within the
		// containing CUDA warp. With multiple groups per warp (SUBWARP < 32),
		// each group masks the warp-wide ballot to its own bit-range.
		const int      lane_in_warp  = static_cast<int>(threadIdx.x) % 32;
		const int      group_in_warp = lane_in_warp / SUBWARP;
		const unsigned subwarp_lanes = (SUBWARP == 32) ? 0xFFFFFFFFu
		                                               : ((1u << SUBWARP) - 1u);
		const unsigned subwarp_mask  = subwarp_lanes << (group_in_warp * SUBWARP);
		__syncwarp(subwarp_mask);

		// Outer ja loop. Inner k loop runs k_iters times *uniformly* across
		// all SUBWARP lanes of the group (k_lane = k_lo + it*SUBWARP + lane,
		// guarded by `in_range = k_lane < k_hi`). This keeps __ballot_sync
		// and __syncwarp(subwarp_mask) calls reachable by every lane on
		// every iteration — fixing the divergent-loop-exit bug in v1.
		// adet_lower_bound returns -1 on miss, otherwise i < AdetToDetOffset[jast+1];
		// so (idxb - AdetToDetOffset[jast]) is in-range whenever idxb >= 0.
		for (uint32_t ja = exidx.SinglesFromAdetOffset[ia]; ja < exidx.SinglesFromAdetOffset[ia + 1]; ja++) {
			uint32_t jast    = exidx.SinglesFromAdetSM[ja];
			uint32_t k_lo    = exidx.SinglesFromBdetOffset[ibst];
			uint32_t k_hi    = exidx.SinglesFromBdetOffset[ibst + 1];
			uint32_t k_iters = (k_hi - k_lo + SUBWARP - 1) / SUBWARP;

			for (uint32_t it = 0; it < k_iters; it++) {
				uint32_t k        = k_lo + it * SUBWARP + lane;
				bool     in_range = (k < k_hi);
				int64_t  idxb     = -1;
				uint32_t jbst     = 0;
				bool    valid    = false;
				if (in_range) {
					jbst  = exidx.SinglesFromBdetSM[k];
					idxb  = tidxmap.adet_lower_bound(jast, jbst, /*start_idx=*/0);
					valid = (idxb >= 0 && jbst == tidxmap.AdetToBdetSM[idxb]);
				}

				// Per-group ballot (only this group's SUBWARP lanes
				// participate; sibling groups in the same warp may be at
				// different ja iterations). Per-lane rank within the group
				// is __popc of group-local bits below this lane. Replaces
				// atomicAdd for slot allocation.
				unsigned warp_ballot    = __ballot_sync(subwarp_mask, valid);
				unsigned subwarp_ballot = (warp_ballot & subwarp_mask) >> (group_in_warp * SUBWARP);
				int      rank           = __popc(subwarp_ballot & ((1u << lane) - 1u));
				int      added          = __popc(subwarp_ballot);

				if (valid) {
					int slot = buf_base + s_count[group] + rank;
					s_idxb[slot] = static_cast<uint32_t>(idxb);
					s_ja  [slot] = ja;
					s_k   [slot] = k;
				}
				if (lane == 0) s_count[group] += added;
				__syncwarp(subwarp_mask);

				// Per Jim's drain-fold suggestion: extend the dispatch trigger
				// to also fire on the very last (it, ja) so the post-loop drain
				// block can be eliminated. Loop the dispatch on is_last_inner so
				// the case where post-append count > SUBWARP on the last (it, ja)
				// is also drained — Jim's literal proposal would lose
				// (count - SUBWARP) candidates in that case (one round of
				// SUBWARP-wide dispatch + compact, then kernel exits with the
				// shifted leftovers still unprocessed). The while form runs at
				// most twice on the last iter and once otherwise.
				bool is_last_inner = (it == k_iters - 1) &&
				                     (ja == exidx.SinglesFromAdetOffset[ia + 1] - 1);
				while (s_count[group] >= SUBWARP || (is_last_inner && s_count[group] > 0)) {
					// Dispatch up to SUBWARP TwoExcite() calls in parallel. Lane
					// guard: count may be < SUBWARP on the is_last_inner drain pass.
					if (lane < s_count[group]) {
						int      my_slot = buf_base + lane;
						uint32_t my_idxb = s_idxb[my_slot];
						uint32_t my_ja   = s_ja  [my_slot];
						uint32_t my_k    = s_k   [my_slot];
						uint32_t jdet    = tidxmap.AdetToDetSM[my_idxb];
						ElemT eij = this->TwoExcite(this->det + idet * this->D_size,
						                    exidx.SinglesAdetCrAnSM[my_ja],
						                    exidx.SinglesBdetCrAnSM[my_k],
						                    exidx.SinglesAdetCrAnSM[my_ja + exidx.size_single_adet],
						                    exidx.SinglesBdetCrAnSM[my_k + exidx.size_single_bdet]);
						thread_sum += eij * this->twk[jdet];
					}
					__syncwarp(subwarp_mask);

					// Compact this group's slice. Source [buf_base+SUBWARP,
					// buf_base+2*SUBWARP) and destination [buf_base, buf_base+SUBWARP)
					// are disjoint; direct lane-parallel copy is safe without an
					// intermediate sync. Slots above the new s_count are garbage
					// but invisible — either next probe round overwrites starting
					// at s_count, or kernel exits (is_last_inner drain case). Only
					// lane 0 mutates s_count (decrement-not-assign).
					s_idxb[buf_base + lane] = s_idxb[buf_base + SUBWARP + lane];
					s_ja  [buf_base + lane] = s_ja  [buf_base + SUBWARP + lane];
					s_k   [buf_base + lane] = s_k   [buf_base + SUBWARP + lane];
					if (lane == 0) s_count[group] -= SUBWARP;
					__syncwarp(subwarp_mask);
				}
			}
		}


		ElemT total = WarpReduceSum(temp_sum).Sum(thread_sum);
		if (lane == 0) this->wb[idet] += total;
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
    size_t task_sent = 0;

	if (exidx[0].slide != 0) {
		sbd::gdb::MpiSlide(idxmap, idxmap_storage, tidxmap[active_buf], tidxmap_storage[active_buf], -exidx[0].slide, this->b_comm());
		sbd::MpiSlide(wk, twk[active_buf], -exidx[0].slide, this->b_comm());
	} else {
		tidxmap[active_buf].copy(tidxmap_storage[active_buf], idxmap, idxmap_storage);
		twk[active_buf] = wk;
	}

	if (mpi_rank_t == 0) {
        auto ci = thrust::counting_iterator<size_t>(0);
        thrust::for_each_n(thrust::device, ci, twk[active_buf].size(), Wb_init_kernel(wb, hii, twk[active_buf]));
	}

	MpiSlider<ElemT> slider;
	for (size_t task = 0; task < exidx.size(); task++) {
        if (task_sent == task) {
            if (task_sent != 0) {
                if (slider.Sync()) {
					int t = active_buf;
					active_buf = recv_buf;
					recv_buf = t;
                }
            }

            // exchange asynchronously for later tasks
            for (size_t extask = task_sent; extask < exidx.size() - 1; extask++) {
				int slide = exidx[extask].slide - exidx[extask + 1].slide;
				slider.ExchangeAsync(twk[active_buf], twk[recv_buf], tidxmap[active_buf], tidxmap_storage[active_buf], tidxmap[recv_buf], tidxmap_storage[recv_buf], slide, this->b_comm(), (int)extask);
				task_sent = extask + 1;
				break;
            }
        }

		// single alpha excitations
		MultSingleAlphaKernel kernel_single_alpha(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_single_alpha.set_mpi_size(mpi_rank_h, mpi_size_h);
		launch_mult_for_each_n<SBD_MULT_BLOCK_SIZE>(
		    SBD_GDB_SUBWARP_SIZE * idxmap.size_adet, kernel_single_alpha);

		// double alpha excitations
		MultDoubleAlphaKernel kernel_double_alpha(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_double_alpha.set_mpi_size(mpi_rank_h, mpi_size_h);
		launch_mult_for_each_n<SBD_MULT_BLOCK_SIZE>(
		    SBD_GDB_SUBWARP_SIZE * idxmap.size_adet, kernel_double_alpha);

		// single beta excitations
		MultSingleBetaKernel kernel_single_beta(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_single_beta.set_mpi_size(mpi_rank_h, mpi_size_h);
		launch_mult_for_each_n<SBD_MULT_BLOCK_SIZE>(
		    SBD_GDB_SUBWARP_SIZE * idxmap.size_bdet, kernel_single_beta);

		// double beta excitations
		MultDoubleBetaKernel kernel_double_beta(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_double_beta.set_mpi_size(mpi_rank_h, mpi_size_h);
		launch_mult_for_each_n<SBD_MULT_BLOCK_SIZE>(
		    SBD_GDB_SUBWARP_SIZE * idxmap.size_bdet, kernel_double_beta);

		// alpha-beta excitations
		MultAlphaBetaKernel kernel_alpha_beta(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_alpha_beta.set_mpi_size(mpi_rank_h, mpi_size_h);
		launch_mult_for_each_n<SBD_MULT_BLOCK_SIZE>(
		    SBD_GDB_SUBWARP_SIZE * idxmap.size_adet, kernel_alpha_beta);
	} // end task for loop

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
		if (i % this->mpi_size_h == this->mpi_rank_h) {
	        size_t* Det = this->det + i * this->D_size;
            hii[i] = this->ZeroExcite(Det);
		}
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
