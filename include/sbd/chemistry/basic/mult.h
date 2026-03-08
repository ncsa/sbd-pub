/**
@file sbd/chemistry/base/mult.h
@brief Function to perform Hamiltonian operation for twist-basis parallelization scheme
*/
#ifndef SBD_CHEMISTRY_MULT_BASE_H
#define SBD_CHEMISTRY_MULT_BASE_H

#include <chrono>
#include <cstdio>

namespace sbd
{

template <typename ElemT>
class MultBase {
protected:
    size_t bit_length_;
    size_t norbs_;
    size_t D_size_;
    MPI_Comm h_comm_;
	MPI_Comm b_comm_;
	MPI_Comm t_comm_;
public:
    MultBase() {}

    MultBase(size_t bit_length, size_t norbs, MPI_Comm h_comm, MPI_Comm b_comm, MPI_Comm t_comm)
        : bit_length(bit_length), norbs(norbs), h_comm(h_comm), b_comm(b_comm), t_comm(t_comm)
    {
        D_size_ = (2 * norbs_ + bit_length_ - 1) / bit_length_;
    }

    inline size_t bit_length(void) const
    {
        return bit_length_;
    }
    inline size_t norbs(void) const
    {
        return norbs_;
    }
    inline size_t D_size(void) const
    {
        return D_size_;
    }
    inline MPI_Comm h_comm(void) const
    {
        return h_comm_;
    }
    inline MPI_Comm b_comm(void) const
    {
        return b_comm_;
    }
    inline MPI_Comm t_comm(void) const
    {
        return t_comm_;
    }

#ifdef SBD_THRUST
    virtual void run(const thrust::device_vector<ElemT> &hii,
                    const thrust::device_vector<ElemT> &Wk,
                    thrust::device_vector<ElemT> &Wb) = 0;
#else
    virtual void run(const std::vector<ElemT> & hii,
            	    const std::vector<ElemT> & Wk,
	                std::vector<ElemT> & Wb) = 0;
#endif
};


}

#endif
