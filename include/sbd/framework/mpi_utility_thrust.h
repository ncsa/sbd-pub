/// This is a part of qscd
/**
@file mpi_utility_thrust.h
@brief tools for mpi parallelization
 */

#ifndef SBD_FRAMEWORK_MPI_UTILITY_THRUST_H
#define SBD_FRAMEWORK_MPI_UTILITY_THRUST_H

#include "mpi.h"

#include "sbd/framework/nvtx.h"

#ifdef SBD_USE_NCCL
#include <nccl.h>
#include <type_traits>

template <typename T>
struct NcclDataType;

template <>
struct NcclDataType<float> {
    static constexpr ncclDataType_t value = ncclFloat;
};

template <>
struct NcclDataType<double> {
    static constexpr ncclDataType_t value = ncclDouble;
};

template <>
struct NcclDataType<int> {
    static constexpr ncclDataType_t value = ncclInt;
};

template <>
struct NcclDataType<int64_t> {
    static constexpr ncclDataType_t value = ncclInt64;
};

#define CHECK_NCCL(cmd) \
    do {                                                        \
        ncclResult_t r = cmd;                                   \
        if (r != ncclSuccess) {                                 \
            fprintf(stderr, "NCCL error %s at %s:%d\n",         \
                    ncclGetErrorString(r), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (0)

#define CHECK_CUDA(cmd) \
    do {                                                        \
        cudaError_t e = cmd;                                    \
        if (e != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n",         \
                    cudaGetErrorString(e), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (0)
#endif // #ifdef SBD_USE_NCCL

namespace sbd
{

#ifdef SBD_THRUST_SAFE_MPI_ALLREDUCE

template <typename ElemT>
void MpiAllreduce(thrust::device_vector<ElemT> &A, MPI_Op op, MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("MpiAllreduce", __LINE__);
    // Calling MPI functions directly on the `device_vector` sometimes caused instability,
    // so copy them to the host temporarily.
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    std::vector<ElemT> h_send(A.size()), h_recv(A.size());
    thrust::copy(A.begin(), A.end(), h_send.begin());
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Allreduce(h_send.data(), h_recv.data(), static_cast<int>(h_send.size()), DataT, op, comm);
    }
    thrust::copy(h_recv.begin(), h_recv.end(), A.begin());
}

#else

template <typename ElemT>
void MpiAllreduce(thrust::device_vector<ElemT> &A, MPI_Op op, MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("MpiAllreduce", __LINE__);
    std::cout << "   TEST MpiAllreduce" << std::endl;
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
#if 1
    thrust::device_vector<ElemT> B(A);
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Allreduce((ElemT *)thrust::raw_pointer_cast(B.data()), (ElemT *)thrust::raw_pointer_cast(A.data()), A.size(), DataT, op, comm);
    }
#else
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce (MPI_IN_PLACE)", 0);
        MPI_Allreduce(MPI_IN_PLACE, (ElemT *)thrust::raw_pointer_cast(A.data()), A.size(), DataT, op, comm);
    }
#endif
}

#endif // SBD_THRUST_SAFE_MPI_ALLREDUCE

template <typename ElemT>
void _MpiSlide(const ElemT* A,
    thrust::device_vector<ElemT>& B,
    size_t sizeA,
    int slide,
    MPI_Comm comm)
{
    int mpi_rank;
    MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size;
    MPI_Comm_size(comm,&mpi_size);
    int mpi_dest   = (mpi_size+mpi_rank+slide) % mpi_size;
    int mpi_source = (mpi_size+mpi_rank-slide) % mpi_size;

    std::vector<MPI_Request> req_size(2);
    std::vector<MPI_Status> sta_size(2);
    std::vector<size_t> size_send(1);
    std::vector<size_t> size_recv(1);
    size_send[0] = sizeA;
    {
        SBD_NVTX_RANGE_COLOR("MPI_Isend", 0);
        MPI_Isend(size_send.data(),1,SBD_MPI_SIZE_T,mpi_dest,0,comm,&req_size[0]);
    }
    {
        SBD_NVTX_RANGE_COLOR("MPI_Irecv", 0);
        MPI_Irecv(size_recv.data(),1,SBD_MPI_SIZE_T,mpi_source,0,comm,&req_size[1]);
    }
    {
        SBD_NVTX_RANGE_COLOR("MPI_Waitall", 0);
        MPI_Waitall(2,req_size.data(),sta_size.data());
    }

    size_t send_size = size_send[0];
    size_t recv_size = size_recv[0];
    B.resize(recv_size);
    std::vector<MPI_Request> req_data(2);
    std::vector<MPI_Status> sta_data(2);

    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    if( send_size != 0 ) {
        SBD_NVTX_RANGE_COLOR("MPI_Isend", 0);
        MPI_Isend(A,send_size,DataT,mpi_dest,1,comm,&req_data[0]);
    }
    if( recv_size != 0 ) {
        SBD_NVTX_RANGE_COLOR("MPI_Irecv", 0);
        MPI_Irecv((ElemT*)thrust::raw_pointer_cast(B.data()),recv_size,DataT,mpi_source,1,comm,&req_data[1]);
    }

    if( send_size != 0 && recv_size != 0 ) {
        SBD_NVTX_RANGE_COLOR("MPI_Waitall", 0);
        MPI_Waitall(2,req_data.data(),sta_data.data());
    } else if ( send_size != 0 && recv_size == 0 ) {
        SBD_NVTX_RANGE_COLOR("MPI_Waitall", 0);
        MPI_Waitall(1,&req_data[0],&sta_data[0]);
    } else if ( send_size == 0 && recv_size != 0 ) {
        SBD_NVTX_RANGE_COLOR("MPI_Waitall", 0);
        MPI_Waitall(1,&req_data[1],&sta_data[1]);
    }
}

template <typename ElemT>
void MpiSlide(const thrust::device_vector<ElemT>& A,
    thrust::device_vector<ElemT>& B,
    int slide,
    MPI_Comm comm)
{
    _MpiSlide((ElemT*)thrust::raw_pointer_cast(A.data()), B, A.size(), slide, comm);
}

template <typename ElemT>
void MpiSlide(const std::vector<ElemT>& A,
    thrust::device_vector<ElemT>& B,
    int slide,
    MPI_Comm comm)
{
    _MpiSlide(A.data(), B, A.size(), slide, comm);
}



template <>
void MpiSlide(const thrust::device_vector<size_t> & A,
    thrust::device_vector<size_t> & B,
    int slide,
    MPI_Comm comm)
{
    int mpi_rank;
    MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size;
    MPI_Comm_size(comm,&mpi_size);
    int mpi_dest   = (mpi_size+mpi_rank+slide) % mpi_size;
    int mpi_source = (mpi_size+mpi_rank-slide) % mpi_size;

    std::vector<MPI_Request> req_size(2);
    std::vector<MPI_Status> sta_size(2);
    std::vector<size_t> size_send(1);
    std::vector<size_t> size_recv(1);
    size_send[0] = A.size();
    MPI_Isend(size_send.data(),1,SBD_MPI_SIZE_T,mpi_dest,0,comm,&req_size[0]);
    MPI_Irecv(size_recv.data(),1,SBD_MPI_SIZE_T,mpi_source,0,comm,&req_size[1]);
    MPI_Waitall(2,req_size.data(),sta_size.data());

    size_t send_size = size_send[0];
    size_t recv_size = size_recv[0];
    B.resize(recv_size);
    std::vector<MPI_Request> req_data(2);
    std::vector<MPI_Status> sta_data(2);

    MPI_Datatype DataT = SBD_MPI_SIZE_T;
    if( send_size != 0 ) {
        MPI_Isend((size_t*)thrust::raw_pointer_cast(A.data()),send_size,DataT,mpi_dest,1,comm,&req_data[0]);
    }
    if( recv_size != 0 ) {
        MPI_Irecv((size_t*)thrust::raw_pointer_cast(B.data()),recv_size,DataT,mpi_source,1,comm,&req_data[1]);
    }

    if( send_size != 0 && recv_size != 0 ) {
        MPI_Waitall(2,req_data.data(),sta_data.data());
    } else if ( send_size != 0 && recv_size == 0 ) {
        MPI_Waitall(1,&req_data[0],&sta_data[0]);
    } else if ( send_size == 0 && recv_size != 0 ) {
        MPI_Waitall(1,&req_data[1],&sta_data[1]);
    }
}



template <typename ElemT>
void _Mpi2dSlide(const ElemT* A,
                thrust::device_vector<ElemT> &B,
                size_t sizeA,
                int x_size,
                int y_size,
                int x_slide,
                int y_slide,
                MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("_Mpi2dSlide", __LINE__);
    // Assuming mpi_rank = x_rank * y_size + y_rank;

    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);
    int mpi_size;
    MPI_Comm_size(comm, &mpi_size);

    int x_rank = mpi_rank / y_size;
    int y_rank = mpi_rank % y_size;

    int x_dist = (x_rank + x_slide + x_size) % x_size;
    int y_dist = (y_rank + y_slide + y_size) % y_size;
    int mpi_dist = x_dist * y_size + y_dist;

    int x_source = (x_rank - x_slide + x_size) % x_size;
    int y_source = (y_rank - y_slide + y_size) % y_size;
    int mpi_source = x_source * y_size + y_source;

#ifdef SBD_DEBUG_MPI_UTILITY
    std::cout << " Mpi2dSlide at rank " << mpi_rank << " = (" << x_rank << "," << y_rank
                << "): distination rank = " << mpi_dist << " = (" << x_dist << "," << y_dist
                << "), source rank = " << mpi_source << " = (" << x_source << "," << y_source
                << ")" << std::endl;
#endif
    std::vector<MPI_Request> req_size(2);
    std::vector<MPI_Status> sta_size(2);
    std::vector<size_t> size_send(1);
    std::vector<size_t> size_recv(1);
    size_send[0] = sizeA;

    MPI_Isend(size_send.data(), 1, SBD_MPI_SIZE_T,
                mpi_dist, 0, comm, &req_size[0]);
    MPI_Irecv(size_recv.data(), 1, SBD_MPI_SIZE_T,
                mpi_source, 0, comm, &req_size[1]);
    MPI_Waitall(2, req_size.data(), sta_size.data());

    size_t send_size = size_send[0];
    size_t recv_size = size_recv[0];
    B.resize(recv_size);
    std::vector<MPI_Request> req_data(2);
    std::vector<MPI_Status> sta_data(2);

    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    if (send_size != 0) {
        MPI_Isend(A, send_size, DataT, mpi_dist, 1, comm, &req_data[0]);
    }
    if (recv_size != 0) {
        MPI_Irecv((ElemT*)thrust::raw_pointer_cast(B.data()), recv_size, DataT, mpi_source, 1, comm, &req_data[1]);
    }

    if (send_size != 0 && recv_size != 0) {
        MPI_Waitall(2, req_data.data(), sta_data.data());
    }
    else if (send_size != 0 && recv_size == 0) {
        MPI_Waitall(1, &req_data[0], &sta_data[0]);
    }
    else if (send_size == 0 && recv_size != 0) {
        MPI_Waitall(1, &req_data[1], &sta_data[1]);
    }
}

template <typename ElemT>
void Mpi2dSlide(const thrust::device_vector<ElemT> &A,
                thrust::device_vector<ElemT> &B,
                int x_size,
                int y_size,
                int x_slide,
                int y_slide,
                MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("Mpi2dSlide", __LINE__);
    _Mpi2dSlide((ElemT*)thrust::raw_pointer_cast(A.data()), B, A.size(),
                      x_size, y_size, x_slide, y_slide, comm);
}

template <typename ElemT>
void Mpi2dSlide(const std::vector<ElemT> &A,
                thrust::device_vector<ElemT> &B,
                int x_size,
                int y_size,
                int x_slide,
                int y_slide,
                MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("Mpi2dSlide", __LINE__);
    _Mpi2dSlide(A.data(), B, A.size(),
                      x_size, y_size, x_slide, y_slide, comm);
}

template <typename ElemT>
class Mpi2dSlider {
protected:
    MPI_Request req_send;
    MPI_Request req_recv;
    size_t send_size;
    size_t recv_size;
public:
    Mpi2dSlider()
    {
        send_size = 0;
        recv_size = 0;
    }

    void ExchangeAsync(const thrust::device_vector<ElemT> &A,
                thrust::device_vector<ElemT> &B,
                int x_size,
                int y_size,
                int x_slide,
                int y_slide,
                MPI_Comm comm,
                size_t task)
    {
        SBD_NVTX_RANGE_COLOR("ExchangeAsync", __LINE__);
        // Assuming mpi_rank = x_rank * y_size + y_rank;

        int mpi_rank;
        MPI_Comm_rank(comm, &mpi_rank);
        int mpi_size;
        MPI_Comm_size(comm, &mpi_size);

        int x_rank = mpi_rank / y_size;
        int y_rank = mpi_rank % y_size;

        int x_dist = (x_rank + x_slide + x_size) % x_size;
        int y_dist = (y_rank + y_slide + y_size) % y_size;
        int mpi_dist = x_dist * y_size + y_dist;

        int x_source = (x_rank - x_slide + x_size) % x_size;
        int y_source = (y_rank - y_slide + y_size) % y_size;
        int mpi_source = x_source * y_size + y_source;

#ifdef SBD_DEBUG_MPI_UTILITY
        std::cout << " Mpi2dSlide at rank " << mpi_rank << " = (" << x_rank << "," << y_rank
                  << "): distination rank = " << mpi_dist << " = (" << x_dist << "," << y_dist
                  << "), source rank = " << mpi_source << " = (" << x_source << "," << y_source
                  << ")" << std::endl;
#endif
        {
            SBD_NVTX_RANGE_COLOR("MPI_Barrier", 0);
            MPI_Barrier(comm);
        }

        std::vector<MPI_Request> req_size(2);
        std::vector<MPI_Status> sta_size(2);
        std::vector<size_t> size_send(1);
        std::vector<size_t> size_recv(1);
        size_send[0] = A.size();

        MPI_Isend(size_send.data(), 1, SBD_MPI_SIZE_T,
                    mpi_dist, task * 2, comm, &req_size[0]);
        MPI_Irecv(size_recv.data(), 1, SBD_MPI_SIZE_T,
                    mpi_source, task * 2, comm, &req_size[1]);
        MPI_Waitall(2, req_size.data(), sta_size.data());

        send_size = size_send[0];
        recv_size = size_recv[0];

        B.resize(recv_size);

        MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
        if (send_size != 0) {
            SBD_NVTX_RANGE_COLOR("MPI_Isend", 0);
            MPI_Isend((ElemT*)thrust::raw_pointer_cast(A.data()), send_size, DataT, mpi_dist, task * 2 + 1, comm, &req_send);
        }
        if (recv_size != 0) {
            SBD_NVTX_RANGE_COLOR("MPI_Irecv", 0);
            MPI_Irecv((ElemT*)thrust::raw_pointer_cast(B.data()), recv_size, DataT, mpi_source, task * 2 + 1, comm, &req_recv);
        }
    }

    bool Sync(MPI_Comm comm)
    {
        SBD_NVTX_RANGE_COLOR("Sync", __LINE__);
        bool recv = false;
        if (send_size > 0) {
            MPI_Status st;

            SBD_NVTX_RANGE_COLOR("MPI_Wait", 0);
            MPI_Wait(&req_send, &st);
        }
        if (recv_size > 0) {
            MPI_Status st;

            SBD_NVTX_RANGE_COLOR("MPI_Wait", 0);
            MPI_Wait(&req_recv, &st);
            recv = true;
        }
        {
            SBD_NVTX_RANGE_COLOR("MPI_Barrier", 0);
            MPI_Barrier(comm);
        }

        send_size = 0;
        recv_size = 0;
        return recv;
    }
};

#ifdef SBD_USE_NCCL
void init_nccl_comm(ncclComm_t *nccl_comm, MPI_Comm mpi_comm)
{
    SBD_NVTX_RANGE_COLOR("init_nccl_comm", __LINE__);
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    {
        SBD_NVTX_RANGE_COLOR("MPI_Bcast", 0);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);
    }
    int mpi_size; MPI_Comm_size(mpi_comm, &mpi_size);
    int mpi_rank; MPI_Comm_rank(mpi_comm, &mpi_rank);
    CHECK_NCCL(ncclCommInitRank(nccl_comm, mpi_size, id, mpi_rank));
}

template <typename ElemT>
void nccl_allreduce(thrust::device_vector<ElemT> &A, ncclRedOp_t nccl_op, ncclComm_t nccl_comm, cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("nccl_allreduce", __LINE__);
    ElemT *buff = thrust::raw_pointer_cast(A.data());
    size_t count = A.size();
    CHECK_NCCL(ncclAllReduce(buff, buff, count, NcclDataType<ElemT>::value, nccl_op, nccl_comm, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename ElemT>
void nccl_allreduce2(thrust::device_vector<ElemT>& A,
                     thrust::device_vector<ElemT>& B,
                     ncclRedOp_t nccl_op,
                     ncclComm_t nccl_comm,
                     void* ws_ptr = nullptr,
                     cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("nccl_allreduce2", __LINE__);
    const size_t nA = A.size();
    const size_t nB = B.size();
    const size_t nTotal = nA + nB;
    if (nTotal == 0) return;

    thrust::device_vector<ElemT> ws_local;
    ElemT* packed_ptr = nullptr;
    if (ws_ptr == nullptr) {
        ws_local.resize(nTotal);
        packed_ptr = thrust::raw_pointer_cast(ws_local.data());
    } else {
        packed_ptr = static_cast<ElemT*>(ws_ptr);
    }
    ElemT* A_ptr = nA ? thrust::raw_pointer_cast(A.data()) : nullptr;
    ElemT* B_ptr = nB ? thrust::raw_pointer_cast(B.data()) : nullptr;

    // pack
    if (nA > 0) {
        CHECK_CUDA(cudaMemcpyAsync(packed_ptr, A_ptr, sizeof(ElemT) * nA,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    if (nB > 0) {
        CHECK_CUDA(cudaMemcpyAsync(packed_ptr + nA, B_ptr, sizeof(ElemT) * nB,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    // single allreduce on packed buffer (in-place)
    CHECK_NCCL(
        ncclAllReduce((const void*)packed_ptr, (void*)packed_ptr, nTotal,
                      NcclDataType<ElemT>::value, nccl_op, nccl_comm, stream));

    // unpack
    if (nA > 0) {
        CHECK_CUDA(cudaMemcpyAsync(A_ptr, packed_ptr, sizeof(ElemT) * nA,
                            cudaMemcpyDeviceToDevice, stream));
    }
    if (nB > 0) {
        CHECK_CUDA(cudaMemcpyAsync(B_ptr, packed_ptr + nA, sizeof(ElemT) * nB,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
}
#endif // #ifdef SBD_USE_NCCL

}

#endif
