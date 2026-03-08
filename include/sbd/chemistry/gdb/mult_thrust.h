/**
@file sbd/chemistry/tpb/mult.h
@brief Function to perform Hamiltonian operation for general determinant basis
*/
#ifndef SBD_CHEMISTRY_GDB_MULT_THRUST_H
#define SBD_CHEMISTRY_GDB_MULT_THRUST_H


#include "sbd/framework/mpi_utility_thrust.h"

namespace sbd
{
namespace gdb
{

template <typename ElemT>
class MultGDBThrust : public sbd::MultBase<ElemT> {
protected:
	thrust::device_vector<size_t> idxmap_storage;
	DetIndexMapThrust idxmap;
    thrust::device_vector<size_t> dets_;
    ElemT I0_;
    oneInt_Thrust<ElemT> I1_;
    thrust::device_vector<ElemT> I1_store;
    twoInt_Thrust<ElemT> I2_;
    thrust::device_vector<ElemT> I2_store;
    thrust::device_vector<ElemT> I2_dm;
    thrust::device_vector<ElemT> I2_em;

	std::vector<thrust::device_vector<size_t>> exidx_storage;
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
    __device__ __host__ void operator()(size_t i)
    {
		size_t ibst = idxmap.AdetToBdetSM[i];
		size_t idet = idxmap.AdetToDetSM[i];
		size_t ia = idxmap.AdetIndex[i];
		size_t iast = ia;
		if (idet % this->mpi_size_h != this->mpi_rank_h)
			return;

		if (exidx.SelfFromBdetOffset[ibst] != exidx.SelfFromBdetOffset[ibst + 1]) {
			size_t jbst = exidx.SelfFromBdetSM[exidx.SelfFromBdetOffset[ibst]];
			// single alpha excitations
			for (size_t ja = exidx.SinglesFromAdetOffset[ia]; ja < exidx.SinglesFromAdetOffset[ia + 1]; ja++) {
				size_t jast = exidx.SinglesFromAdetSM[ja];
				int64_t idxa = tidxmap.bdet_lower_bound(jbst, jast);
				if (idxa >= 0) {
					if (jast != tidxmap.BdetToAdetSM[idxa])
						continue;
					size_t jdet = tidxmap.BdetToDetSM[idxa];
					ElemT eij = this->OneExcite(this->det + idet * this->D_size,
											exidx.SinglesAdetCrAnSM[ja],
											exidx.SinglesAdetCrAnSM[ja + exidx.size_single_adet]);
					this->wb[idet] += eij * this->twk[jdet];
				}
			}
		} // if there is same beta string
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
    __device__ __host__ void operator()(size_t i)
    {
		size_t ibst = idxmap.AdetToBdetSM[i];
		size_t idet = idxmap.AdetToDetSM[i];
		size_t ia = idxmap.AdetIndex[i];
		size_t iast = ia;
		if (idet % this->mpi_size_h != this->mpi_rank_h)
			return;

		if (exidx.SelfFromBdetOffset[ibst] != exidx.SelfFromBdetOffset[ibst + 1]) {
			size_t jbst = exidx.SelfFromBdetSM[exidx.SelfFromBdetOffset[ibst]];
			// double alpha excitations
			for (size_t ja = exidx.DoublesFromAdetOffset[ia]; ja < exidx.DoublesFromAdetOffset[ia + 1]; ja++) {
				size_t jast = exidx.DoublesFromAdetSM[ja];
				int64_t idxa = tidxmap.bdet_lower_bound(jbst, jast);
				if (idxa >= 0) {
					if (jast != tidxmap.BdetToAdetSM[idxa])
						continue;
					size_t jdet = tidxmap.BdetToDetSM[idxa];
					ElemT eij = this->TwoExcite(this->det + idet * this->D_size,
											exidx.DoublesAdetCrAnSM[ja],
											exidx.DoublesAdetCrAnSM[ja + exidx.size_double_adet],
											exidx.DoublesAdetCrAnSM[ja + exidx.size_double_adet * 2],
											exidx.DoublesAdetCrAnSM[ja + exidx.size_double_adet * 3]);
					// size_t od;
					this->wb[idet] += eij * this->twk[jdet];
				}
			}
		} // if there is same beta string
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
    __device__ __host__ void operator()(size_t i)
    {
		size_t iast = idxmap.BdetToAdetSM[i];
		size_t idet = idxmap.BdetToDetSM[i];
		size_t ib = idxmap.BdetIndex[i];
		size_t ibst = ib;
		if (idet % this->mpi_size_h != this->mpi_rank_h)
			return;

		if (exidx.SelfFromAdetOffset[iast] != exidx.SelfFromAdetOffset[iast + 1]) {
			size_t jast = exidx.SelfFromAdetSM[exidx.SelfFromAdetOffset[iast]];
			// single alpha excitations
			for (size_t jb = exidx.SinglesFromBdetOffset[ib]; jb < exidx.SinglesFromBdetOffset[ib + 1]; jb++) {
				size_t jbst = exidx.SinglesFromBdetSM[jb];
				int64_t idxb = tidxmap.adet_lower_bound(jast, jbst);
				if (idxb >= 0) {
					if (jbst != tidxmap.AdetToBdetSM[idxb])
						continue;
					size_t jdet = tidxmap.AdetToDetSM[idxb];
					ElemT eij = this->OneExcite(this->det + idet * this->D_size,
											exidx.SinglesBdetCrAnSM[jb],
											exidx.SinglesBdetCrAnSM[jb + exidx.size_single_bdet]);
					this->wb[idet] += eij * this->twk[jdet];
				}
			}
		} // if there is same beta string
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
    __device__ __host__ void operator()(size_t i)
    {
		size_t iast = idxmap.BdetToAdetSM[i];
		size_t idet = idxmap.BdetToDetSM[i];
		size_t ib = idxmap.BdetIndex[i];
		size_t ibst = ib;
		if (idet % this->mpi_size_h != this->mpi_rank_h)
			return;

		if (exidx.SelfFromAdetOffset[iast] != exidx.SelfFromAdetOffset[iast + 1]) {
			size_t jast = exidx.SelfFromAdetSM[exidx.SelfFromAdetOffset[iast]];
			// double alpha excitations
			for (size_t jb = exidx.DoublesFromBdetOffset[ib]; jb < exidx.DoublesFromBdetOffset[ib + 1]; jb++) {
				size_t jbst = exidx.DoublesFromBdetSM[jb];
				int64_t idxb = tidxmap.adet_lower_bound(jast, jbst);
				if (idxb >= 0) {
					if (jbst != tidxmap.AdetToBdetSM[idxb])
						continue;
					size_t jdet = tidxmap.AdetToDetSM[idxb];
					ElemT eij = this->TwoExcite(this->det + idet * this->D_size,
											exidx.DoublesBdetCrAnSM[jb],
											exidx.DoublesBdetCrAnSM[jb + exidx.size_double_bdet],
											exidx.DoublesBdetCrAnSM[jb + exidx.size_double_bdet * 2],
											exidx.DoublesBdetCrAnSM[jb + exidx.size_double_bdet * 3]);
					// size_t od;
					this->wb[idet] += eij * this->twk[jdet];
				}
			}
		} // if there is same beta string
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
    __device__ __host__ void operator()(size_t i)
    {
		size_t ibst = idxmap.AdetToBdetSM[i];
		size_t idet = idxmap.AdetToDetSM[i];
		size_t ia = idxmap.AdetIndex[i];
		size_t iast = ia;
		if (idet % this->mpi_size_h != this->mpi_rank_h)
			return;

		// alpha-beta two-particle excitations
		for (size_t ja = exidx.SinglesFromAdetOffset[ia]; ja < exidx.SinglesFromAdetOffset[ia + 1]; ja++) {
			size_t jast = exidx.SinglesFromAdetSM[ja];
			size_t start_idx = 0;
			size_t end_idx = tidxmap.AdetToDetOffset[jast + 1] - tidxmap.AdetToDetOffset[jast];
			for (size_t k = exidx.SinglesFromBdetOffset[ibst]; k < exidx.SinglesFromBdetOffset[ibst + 1]; k++) {
				size_t jbst = exidx.SinglesFromBdetSM[k];
				if (start_idx >= end_idx)
					break;
				int64_t idxb = tidxmap.adet_lower_bound(jast, jbst, start_idx);
				if (idxb >= 0) {
					if (jbst != tidxmap.AdetToBdetSM[idxb])
						continue;
					start_idx = idxb - tidxmap.AdetToDetOffset[jast];
					if (start_idx < end_idx) {
						size_t jdet = tidxmap.AdetToDetSM[idxb];
						ElemT eij = this->TwoExcite(this->det + idet * this->D_size,
												exidx.SinglesAdetCrAnSM[ja],
												exidx.SinglesBdetCrAnSM[k],
												exidx.SinglesAdetCrAnSM[ja + exidx.size_single_adet],
												exidx.SinglesBdetCrAnSM[k + exidx.size_single_bdet]);
						// size_t odiff;
						// ElemT eij = Hij(det[idet],tdet[jdet],bit_length,norb,I0,I1,I2,odiff);
						this->wb[idet] += eij * this->twk[jdet];
					}
				}
			}
		}
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
	thrust::device_vector<size_t> tidxmap_storage[2];
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

		auto ci_sa = thrust::counting_iterator<size_t>(0);
		thrust::for_each_n(thrust::device, ci_sa, idxmap.size_adet, kernel_single_alpha);

		// double alpha excitations
		MultDoubleAlphaKernel kernel_double_alpha(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_double_alpha.set_mpi_size(mpi_rank_h, mpi_size_h);

		auto ci_da = thrust::counting_iterator<size_t>(0);
		thrust::for_each_n(thrust::device, ci_da, idxmap.size_adet, kernel_double_alpha);

		// single beta excitations
		MultSingleBetaKernel kernel_single_beta(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_single_beta.set_mpi_size(mpi_rank_h, mpi_size_h);

		auto ci_sb = thrust::counting_iterator<size_t>(0);
		thrust::for_each_n(thrust::device, ci_sb, idxmap.size_bdet, kernel_single_beta);

		// double beta excitations
		MultDoubleBetaKernel kernel_double_beta(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_double_beta.set_mpi_size(mpi_rank_h, mpi_size_h);

		auto ci_db = thrust::counting_iterator<size_t>(0);
		thrust::for_each_n(thrust::device, ci_db, idxmap.size_bdet, kernel_double_beta);

		// alpha-beta excitations
		MultAlphaBetaKernel kernel_alpha_beta(wb, twk[active_buf], *this, idxmap, tidxmap[active_buf], exidx[task]);
		kernel_alpha_beta.set_mpi_size(mpi_rank_h, mpi_size_h);

		auto ci_ab = thrust::counting_iterator<size_t>(0);
		thrust::for_each_n(thrust::device, ci_ab, idxmap.size_adet, kernel_alpha_beta);
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

    hii.resize(dets_.size(), ElemT(0.0));

    MakeQChamDiagTermKernel kernel(hii, *this);
	kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
    auto ci = thrust::counting_iterator<size_t>(0);
    thrust::for_each_n(thrust::device, ci, dets_.size(), kernel);
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
