/**
@file sbd/chemistry/gdb/helpers.h
@brief Helper array to construct Hamiltonian for general determinant basis for Thrust
 */
#ifndef SBD_CHEMISTRY_GDB_HELPER_THRUST_H
#define SBD_CHEMISTRY_GDB_HELPER_THRUST_H

#include "sbd/framework/type_def.h"
#include "sbd/framework/mpi_utility.h"

namespace sbd
{
namespace gdb
{

/**
       DetBasisMapper is an array to find relation between dets.
*/
class DetIndexMapThrust
{
public:
    uint32_t* AdetToDetOffset;
    uint32_t* BdetToDetOffset;
    uint32_t* AdetIndex;
    uint32_t* BdetIndex;
    uint32_t* AdetToBdetSM;
    uint32_t* AdetToDetSM;
    uint32_t* BdetToAdetSM;
    uint32_t* BdetToDetSM;
    uint32_t size_adet = 0;
    uint32_t size_bdet = 0;

    DetIndexMapThrust() {}

    DetIndexMapThrust(const DetIndexMapThrust& other)
    {
        AdetToDetOffset = other.AdetToDetOffset;
        BdetToDetOffset = other.BdetToDetOffset;
        AdetIndex = other.AdetIndex;
        BdetIndex = other.BdetIndex;
        AdetToBdetSM = other.AdetToBdetSM;
        AdetToDetSM = other.AdetToDetSM;
        BdetToAdetSM = other.BdetToAdetSM;
        BdetToDetSM = other.BdetToDetSM;
        size_adet = other.size_adet;
        size_bdet = other.size_bdet;
    }

    DetIndexMapThrust(thrust::device_vector<uint32_t>& storage, const DetIndexMap& idxmap)
    {
        const size_t n_adet = idxmap.AdetToDetLen.size();
        const size_t n_bdet = idxmap.BdetToDetLen.size();

        assert(n_adet <= UINT32_MAX);
        assert(n_bdet <= UINT32_MAX);

        std::vector<uint32_t> offset_adet(n_adet + 1);
        std::vector<uint32_t> offset_bdet(n_bdet + 1);
        for (size_t i = 0; i < n_adet; i++) {
            offset_adet[i] = size_adet;
            assert(idxmap.AdetToDetLen[i] <= UINT32_MAX - (size_t)size_adet);
            size_adet += static_cast<uint32_t>(idxmap.AdetToDetLen[i]);
        }
        offset_adet[n_adet] = size_adet;

        for (size_t i = 0; i < n_bdet; i++) {
            offset_bdet[i] = size_bdet;
            assert(idxmap.BdetToDetLen[i] <= UINT32_MAX - (size_t)size_bdet);
            size_bdet += static_cast<uint32_t>(idxmap.BdetToDetLen[i]);
        }
        offset_bdet[n_bdet] = size_bdet;

        size_t size = (n_adet + 1) + (n_bdet + 1) + (size_t)size_adet * 3 + (size_t)size_bdet * 3;
        storage.resize(size);
        uint32_t* base_memory = thrust::raw_pointer_cast(storage.data());

        size_t count = 0;
        // store offsets
        AdetToDetOffset = base_memory + count;
        thrust::copy_n(offset_adet.begin(), offset_adet.size(), storage.begin() + count);
        count += offset_adet.size();

        BdetToDetOffset = base_memory + count;
        thrust::copy_n(offset_bdet.begin(), offset_bdet.size(), storage.begin() + count);
        count += offset_bdet.size();

        // set Adet, Bdet indices — per-element fill_n, one kernel per segment
        AdetIndex = base_memory + count;
        for (size_t i = 0; i < n_adet; i++) {
            thrust::fill_n(storage.begin() + count + offset_adet[i],
                           idxmap.AdetToDetLen[i], static_cast<uint32_t>(i));
        }
        count += size_adet;

        BdetIndex = base_memory + count;
        for (size_t i = 0; i < n_bdet; i++) {
            thrust::fill_n(storage.begin() + count + offset_bdet[i],
                           idxmap.BdetToDetLen[i], static_cast<uint32_t>(i));
        }
        count += size_bdet;

        // copy SMs: convert size_t CPU data to uint32_t on device
        const size_t sm_size = (size_t)size_adet * 2 + (size_t)size_bdet * 2;
        std::vector<uint32_t> sm_buf(sm_size);
        for (size_t i = 0; i < sm_size; i++) {
            assert(idxmap.storage[i] <= UINT32_MAX);
            sm_buf[i] = static_cast<uint32_t>(idxmap.storage[i]);
        }
        thrust::copy_n(sm_buf.begin(), sm_buf.size(), storage.begin() + count);
        AdetToDetSM = base_memory + count;
        count += size_adet;
        AdetToBdetSM = base_memory + count;
        count += size_adet;
        BdetToDetSM = base_memory + count;
        count += size_bdet;
        BdetToAdetSM = base_memory + count;
    }

    void copy(thrust::device_vector<uint32_t>& storage, const DetIndexMapThrust& idxmap, const thrust::device_vector<uint32_t>& src_storage)
    {
        storage = src_storage;

        uint32_t* base = thrust::raw_pointer_cast(storage.data());
        const uint32_t* src_base = thrust::raw_pointer_cast(src_storage.data());

        AdetToDetOffset = base + (idxmap.AdetToDetOffset - src_base);
        BdetToDetOffset = base + (idxmap.BdetToDetOffset - src_base);
        AdetIndex = base + (idxmap.AdetIndex - src_base);
        BdetIndex = base + (idxmap.BdetIndex - src_base);
        AdetToBdetSM = base + (idxmap.AdetToBdetSM - src_base);
        AdetToDetSM = base + (idxmap.AdetToDetSM - src_base);
        BdetToAdetSM = base + (idxmap.BdetToAdetSM - src_base);
        BdetToDetSM = base + (idxmap.BdetToDetSM - src_base);
        size_adet = idxmap.size_adet;
        size_bdet = idxmap.size_bdet;
    }

    inline __device__ __host__ std::pair<bool, uint32_t> adet_lower_bound(uint32_t jast, uint32_t jbst)
    {
        uint32_t is = AdetToDetOffset[jast];
        uint32_t ie = AdetToDetOffset[jast + 1];
        while (is != ie) {
            uint32_t i = (is + ie) / 2;
            if (AdetToBdetSM[i] < jbst)
                is = i + 1;
            else
                ie = i;
        }
        return {ie != AdetToDetOffset[jast + 1] && AdetToBdetSM[ie] == jbst, ie};
    }

    inline __device__ __host__ std::pair<bool, uint32_t> bdet_lower_bound(uint32_t jbst, uint32_t jast)
    {
        uint32_t is = BdetToDetOffset[jbst];
        uint32_t ie = BdetToDetOffset[jbst + 1];
        while (is != ie) {
            uint32_t i = (is + ie) / 2;
            if (BdetToAdetSM[i] < jast)
                is = i + 1;
            else
                ie = i;
        }
        return {ie != BdetToDetOffset[jbst + 1] && BdetToAdetSM[ie] == jast, ie};
    }


};

/**
       Labels connected
*/
class ExcitationLookupThrust
{
public:
    int slide;
    uint32_t* SelfFromAdetOffset;
    uint32_t* SelfFromBdetOffset;
    uint32_t* SinglesFromAdetOffset;
    uint32_t* SinglesFromBdetOffset;
    uint32_t* DoublesFromAdetOffset;
    uint32_t* DoublesFromBdetOffset;
    uint32_t* SelfFromAdetSM;
    uint32_t* SelfFromBdetSM;
    uint32_t* SinglesFromAdetSM;
    int* SinglesAdetCrAnSM;
    uint32_t* SinglesFromBdetSM;
    int* SinglesBdetCrAnSM;
    uint32_t* DoublesFromAdetSM;
    int* DoublesAdetCrAnSM;
    uint32_t* DoublesFromBdetSM;
    int* DoublesBdetCrAnSM;
    uint32_t size_self_adet = 0;
    uint32_t size_self_bdet = 0;
    uint32_t size_single_adet = 0;
    uint32_t size_single_bdet = 0;
    uint32_t size_double_adet = 0;
    uint32_t size_double_bdet = 0;

    ExcitationLookupThrust() {}

    ExcitationLookupThrust(const ExcitationLookupThrust& other)
    {
        slide = other.slide;
        SelfFromAdetOffset = other.SelfFromAdetOffset;
        SelfFromBdetOffset = other.SelfFromBdetOffset;
        SinglesFromAdetOffset = other.SinglesFromAdetOffset;
        SinglesFromBdetOffset = other.SinglesFromBdetOffset;
        DoublesFromAdetOffset = other.DoublesFromAdetOffset;
        DoublesFromBdetOffset = other.DoublesFromBdetOffset;
        SelfFromAdetSM = other.SelfFromAdetSM;
        SelfFromBdetSM = other.SelfFromBdetSM;
        SinglesFromAdetSM = other.SinglesFromAdetSM;
        SinglesAdetCrAnSM = other.SinglesAdetCrAnSM;
        SinglesFromBdetSM = other.SinglesFromBdetSM;
        SinglesBdetCrAnSM = other.SinglesBdetCrAnSM;
        DoublesFromAdetSM = other.DoublesFromAdetSM;
        DoublesAdetCrAnSM = other.DoublesAdetCrAnSM;
        DoublesFromBdetSM = other.DoublesFromBdetSM;
        DoublesBdetCrAnSM = other.DoublesBdetCrAnSM;

        size_self_adet = other.size_self_adet;
        size_self_bdet = other.size_self_bdet;
        size_single_adet = other.size_single_adet;
        size_single_bdet = other.size_single_bdet;
        size_double_adet = other.size_double_adet;
        size_double_bdet = other.size_double_bdet;
    }

    ExcitationLookupThrust(thrust::device_vector<uint32_t>& storage, thrust::device_vector<int>& cran_storage, const ExcitationLookup& exidx)
    {
        size_t size = 0;

        slide = exidx.slide;

        std::vector<uint32_t> offset_self_adet(exidx.SelfFromAdetLen.size() + 1);
        for(size_t i=0; i < exidx.SelfFromAdetLen.size(); i++) {
            offset_self_adet[i] = size_self_adet;
            assert(exidx.SelfFromAdetLen[i] <= UINT32_MAX - (size_t)size_self_adet);
            size_self_adet += static_cast<uint32_t>(exidx.SelfFromAdetLen[i]);
            size++;
        }
        offset_self_adet[exidx.SelfFromAdetLen.size()] = size_self_adet;
        size += (size_t)size_self_adet + 1;

        std::vector<uint32_t> offset_self_bdet(exidx.SelfFromBdetLen.size() + 1);
        for(size_t i=0; i < exidx.SelfFromBdetLen.size(); i++) {
            offset_self_bdet[i] = size_self_bdet;
            assert(exidx.SelfFromBdetLen[i] <= UINT32_MAX - (size_t)size_self_bdet);
            size_self_bdet += static_cast<uint32_t>(exidx.SelfFromBdetLen[i]);
            size++;
        }
        offset_self_bdet[exidx.SelfFromBdetLen.size()] = size_self_bdet;
        size += (size_t)size_self_bdet + 1;

        std::vector<uint32_t> offset_single_adet(exidx.SinglesFromAdetLen.size() + 1);
        for(size_t i=0; i < exidx.SinglesFromAdetLen.size(); i++) {
            offset_single_adet[i] = size_single_adet;
            assert(exidx.SinglesFromAdetLen[i] <= UINT32_MAX - (size_t)size_single_adet);
            size_single_adet += static_cast<uint32_t>(exidx.SinglesFromAdetLen[i]);
            size++;
        }
        offset_single_adet[exidx.SinglesFromAdetLen.size()] = size_single_adet;
        size += (size_t)size_single_adet + 1;

        std::vector<uint32_t> offset_double_adet(exidx.DoublesFromAdetLen.size() + 1);
        for(size_t i=0; i < exidx.DoublesFromAdetLen.size(); i++) {
            offset_double_adet[i] = size_double_adet;
            assert(exidx.DoublesFromAdetLen[i] <= UINT32_MAX - (size_t)size_double_adet);
            size_double_adet += static_cast<uint32_t>(exidx.DoublesFromAdetLen[i]);
            size++;
        }
        offset_double_adet[exidx.DoublesFromAdetLen.size()] = size_double_adet;
        size += (size_t)size_double_adet + 1;

        std::vector<uint32_t> offset_single_bdet(exidx.SinglesFromBdetLen.size() + 1);
        for(size_t i=0; i < exidx.SinglesFromBdetLen.size(); i++) {
            offset_single_bdet[i] = size_single_bdet;
            assert(exidx.SinglesFromBdetLen[i] <= UINT32_MAX - (size_t)size_single_bdet);
            size_single_bdet += static_cast<uint32_t>(exidx.SinglesFromBdetLen[i]);
            size++;
        }
        offset_single_bdet[exidx.SinglesFromBdetLen.size()] = size_single_bdet;
        size += (size_t)size_single_bdet + 1;

        std::vector<uint32_t> offset_double_bdet(exidx.DoublesFromBdetLen.size() + 1);
        for(size_t i=0; i < exidx.DoublesFromBdetLen.size(); i++) {
            offset_double_bdet[i] = size_double_bdet;
            assert(exidx.DoublesFromBdetLen[i] <= UINT32_MAX - (size_t)size_double_bdet);
            size_double_bdet += static_cast<uint32_t>(exidx.DoublesFromBdetLen[i]);
            size++;
        }
        offset_double_bdet[exidx.DoublesFromBdetLen.size()] = size_double_bdet;
        size += (size_t)size_double_bdet + 1;

        storage.resize(size);
        uint32_t* base_memory = thrust::raw_pointer_cast(storage.data());

        size_t count = 0;
        // store offset
        SelfFromAdetOffset = base_memory + count;
        thrust::copy_n(offset_self_adet.begin(), offset_self_adet.size(), storage.begin() + count);
        count += offset_self_adet.size();

        SelfFromBdetOffset = base_memory + count;
        thrust::copy_n(offset_self_bdet.begin(), offset_self_bdet.size(), storage.begin() + count);
        count += offset_self_bdet.size();

        SinglesFromAdetOffset = base_memory + count;
        thrust::copy_n(offset_single_adet.begin(), offset_single_adet.size(), storage.begin() + count);
        count += offset_single_adet.size();

        DoublesFromAdetOffset = base_memory + count;
        thrust::copy_n(offset_double_adet.begin(), offset_double_adet.size(), storage.begin() + count);
        count += offset_double_adet.size();

        SinglesFromBdetOffset = base_memory + count;
        thrust::copy_n(offset_single_bdet.begin(), offset_single_bdet.size(), storage.begin() + count);
        count += offset_single_bdet.size();

        DoublesFromBdetOffset = base_memory + count;
        thrust::copy_n(offset_double_bdet.begin(), offset_double_bdet.size(), storage.begin() + count);
        count += offset_double_bdet.size();

        // copy SMs (exidx.storage is size_t; convert to uint32_t with overflow assertions)
        size_t sm_count = size - count;
        std::vector<uint32_t> sm_buf(sm_count);
        for (size_t i = 0; i < sm_count; i++) {
            assert(exidx.storage[i] <= UINT32_MAX);
            sm_buf[i] = static_cast<uint32_t>(exidx.storage[i]);
        }
        thrust::copy_n(sm_buf.begin(), sm_count, storage.begin() + count);
        SelfFromAdetSM = base_memory + count;
        count += size_self_adet;
        SinglesFromAdetSM = base_memory + count;
        count += size_single_adet;
        DoublesFromAdetSM = base_memory + count;
        count += size_double_adet;
        SelfFromBdetSM = base_memory + count;
        count += size_self_bdet;
        SinglesFromBdetSM = base_memory + count;
        count += size_single_bdet;
        DoublesFromBdetSM = base_memory + count;

        // convert CrAn from AoS to SoA
        size_t size_cran = (size_t)size_single_adet * 2 + (size_t)size_double_adet * 4 + (size_t)size_single_bdet * 2 + (size_t)size_double_bdet * 4;
        std::vector<int> buf;
        count = 0;

        cran_storage.resize(size_cran);
        int* base_cran = (int*)thrust::raw_pointer_cast(cran_storage.data());

        SinglesAdetCrAnSM = base_cran + count;
        buf.resize((size_t)size_single_adet * 2);
#pragma omp parallel for
        for(size_t i=0; i < exidx.SinglesFromAdetLen.size(); i++) {
            for (int j=0; j <  exidx.SinglesFromAdetLen[i]; j++) {
                size_t offset = offset_single_adet[i] + j;
                buf[offset] = exidx.SinglesAdetCrAnSM[i][j * 2];
                buf[(size_t)size_single_adet + offset] = exidx.SinglesAdetCrAnSM[i][j * 2 + 1];
            }
        }
        thrust::copy_n(buf.begin(), (size_t)size_single_adet * 2, cran_storage.begin() + count);
        count += (size_t)size_single_adet * 2;


        DoublesAdetCrAnSM = base_cran + count;
        buf.resize((size_t)size_double_adet * 4);
#pragma omp parallel for
        for(size_t i=0; i < exidx.DoublesFromAdetLen.size(); i++) {
            for (int j=0; j <  exidx.DoublesFromAdetLen[i]; j++) {
                size_t offset = offset_double_adet[i] + j;
                buf[offset] = exidx.DoublesAdetCrAnSM[i][j * 4];
                buf[(size_t)size_double_adet + offset] = exidx.DoublesAdetCrAnSM[i][j * 4 + 1];
                buf[(size_t)size_double_adet * 2 + offset] = exidx.DoublesAdetCrAnSM[i][j * 4 + 2];
                buf[(size_t)size_double_adet * 3 + offset] = exidx.DoublesAdetCrAnSM[i][j * 4 + 3];
            }
        }
        thrust::copy_n(buf.begin(), (size_t)size_double_adet * 4, cran_storage.begin() + count);
        count += (size_t)size_double_adet * 4;

        SinglesBdetCrAnSM = base_cran + count;
        buf.resize((size_t)size_single_bdet * 2);
#pragma omp parallel for
        for(size_t i=0; i < exidx.SinglesFromBdetLen.size(); i++) {
            for (int j=0; j <  exidx.SinglesFromBdetLen[i]; j++) {
                size_t offset = offset_single_bdet[i] + j;
                buf[offset] = exidx.SinglesBdetCrAnSM[i][j * 2];
                buf[(size_t)size_single_bdet + offset] = exidx.SinglesBdetCrAnSM[i][j * 2 + 1];
            }
        }
        thrust::copy_n(buf.begin(), (size_t)size_single_bdet * 2, cran_storage.begin() + count);
        count += (size_t)size_single_bdet * 2;

        DoublesBdetCrAnSM = base_cran + count;
        buf.resize((size_t)size_double_bdet * 4);
#pragma omp parallel for
        for(size_t i=0; i < exidx.DoublesFromBdetLen.size(); i++) {
            for (int j=0; j <  exidx.DoublesFromBdetLen[i]; j++) {
                size_t offset = offset_double_bdet[i] + j;
                buf[offset] = exidx.DoublesBdetCrAnSM[i][j * 4];
                buf[(size_t)size_double_bdet + offset] = exidx.DoublesBdetCrAnSM[i][j * 4 + 1];
                buf[(size_t)size_double_bdet * 2 + offset] = exidx.DoublesBdetCrAnSM[i][j * 4 + 2];
                buf[(size_t)size_double_bdet * 3 + offset] = exidx.DoublesBdetCrAnSM[i][j * 4 + 3];
            }
        }
        thrust::copy_n(buf.begin(), (size_t)size_double_bdet * 4, cran_storage.begin() + count);
        count += (size_t)size_double_bdet * 4;
    }


};


void MpiSlide(const DetIndexMapThrust& send_map,
        const thrust::device_vector<uint32_t>& send_storage,
        DetIndexMapThrust& recv_map,
        thrust::device_vector<uint32_t>& recv_storage,
        int slide,
        MPI_Comm comm)
{
    std::vector<size_t> send_offset;
    std::vector<size_t> recv_offset;

    {
        const uint32_t* send_base = thrust::raw_pointer_cast(send_storage.data());
        send_offset.push_back((size_t)(send_map.AdetToDetOffset - send_base));
        send_offset.push_back((size_t)(send_map.BdetToDetOffset - send_base));
        send_offset.push_back((size_t)(send_map.AdetIndex - send_base));
        send_offset.push_back((size_t)(send_map.BdetIndex - send_base));
        send_offset.push_back((size_t)(send_map.AdetToBdetSM - send_base));
        send_offset.push_back((size_t)(send_map.AdetToDetSM - send_base));
        send_offset.push_back((size_t)(send_map.BdetToAdetSM - send_base));
        send_offset.push_back((size_t)(send_map.BdetToDetSM - send_base));
    }
    send_offset.push_back((size_t)send_map.size_adet);
    send_offset.push_back((size_t)send_map.size_bdet);

    sbd::MpiSlide(send_storage, recv_storage, slide, comm);
    sbd::MpiSlide(send_offset, recv_offset, slide, comm);

    uint32_t* base = thrust::raw_pointer_cast(recv_storage.data());
    recv_map.AdetToDetOffset = base + recv_offset[0];
    recv_map.BdetToDetOffset = base + recv_offset[1];
    recv_map.AdetIndex = base + recv_offset[2];
    recv_map.BdetIndex = base + recv_offset[3];
    recv_map.AdetToBdetSM = base + recv_offset[4];
    recv_map.AdetToDetSM = base + recv_offset[5];
    recv_map.BdetToAdetSM = base + recv_offset[6];
    recv_map.BdetToDetSM = base + recv_offset[7];
    recv_map.size_adet = static_cast<uint32_t>(recv_offset[8]);
    recv_map.size_bdet = static_cast<uint32_t>(recv_offset[9]);
}



template <typename ElemT>
class MpiSlider {
protected:
    MPI_Request req_send;
    MPI_Request req_recv;
    MPI_Request req_send_map;
    MPI_Request req_recv_map;
    size_t send_size;
    size_t recv_size;
    size_t send_size_map;
    size_t recv_size_map;
public:
    MpiSlider()
    {
        send_size = 0;
        recv_size = 0;
        send_size_map = 0;
        recv_size_map = 0;
    }

    // Return the ket/map recv sizes recorded by the most recent ExchangeAsync call.
    // Valid after ExchangeAsync returns and before Sync() clears them.
    size_t get_recv_size()     const { return recv_size; }
    size_t get_recv_size_map() const { return recv_size_map; }

    // ExchangeAsync with caller-managed recv buffer capacity.
    //
    // send_ket_size    — actual number of ket elements in send (the caller
    //                    tracks this separately because send.size() may equal
    //                    global_max_ket_size after pre-allocation).
    // send_map_size    — actual number of map elements in send_map_storage.
    //
    // recv / recv_map_storage — both device_vectors, pre-allocated by the
    //                    caller to global_max_ket_size / global_max_map_size
    //                    respectively and kept at that size throughout the
    //                    task loop.  No resize is performed here; MPI_Irecv
    //                    writes directly into the pre-allocated device memory.
    //
    // The caller is responsible for:
    //   1. Before the first call: resize both recv buffers to the global max
    //      (MPI_Allreduce MAX of local sizes), then cudaDeviceSynchronize()
    //      so the fill kernels complete before MPI_Irecv fires.
    //   2. After Sync() returns true: update the tracked ket/map sizes via
    //      get_recv_size() / get_recv_size_map(), then swap active/recv buffers.
    void ExchangeAsync(const thrust::device_vector<ElemT> &send,
                size_t send_ket_size,
                thrust::device_vector<ElemT> &recv,
                const DetIndexMapThrust& send_map,
                const thrust::device_vector<uint32_t> &send_map_storage,
                size_t send_map_size,
                DetIndexMapThrust& recv_map,
                thrust::device_vector<uint32_t> &recv_map_storage,
                int slide,
                MPI_Comm comm,
                int id)
    {
        int mpi_rank;
        MPI_Comm_rank(comm,&mpi_rank);
        int mpi_size;
        MPI_Comm_size(comm,&mpi_size);
        int mpi_dest   = (mpi_size+mpi_rank+slide) % mpi_size;
        int mpi_source = (mpi_size+mpi_rank-slide) % mpi_size;

        // exchange data size to be sent
        std::vector<MPI_Request> req_size(2);
        std::vector<MPI_Status> sta_size(2);
        std::vector<size_t> send_offset(12);
        std::vector<size_t> recv_offset(12);
        // Use the caller-supplied logical sizes, not send.size() /
        // send_map_storage.size(), which may equal global_max after
        // pre-allocation rather than the actual ket/map element count.
        send_offset[0] = send_ket_size;
        send_offset[1] = send_map_size;

        // exchange idxmap offset
        {
            const uint32_t* send_base = thrust::raw_pointer_cast(send_map_storage.data());
            send_offset[2] = (size_t)(send_map.AdetToDetOffset - send_base);
            send_offset[3] = (size_t)(send_map.BdetToDetOffset - send_base);
            send_offset[4] = (size_t)(send_map.AdetIndex - send_base);
            send_offset[5] = (size_t)(send_map.BdetIndex - send_base);
            send_offset[6] = (size_t)(send_map.AdetToBdetSM - send_base);
            send_offset[7] = (size_t)(send_map.AdetToDetSM - send_base);
            send_offset[8] = (size_t)(send_map.BdetToAdetSM - send_base);
            send_offset[9] = (size_t)(send_map.BdetToDetSM - send_base);
        }
        send_offset[10] = (size_t)send_map.size_adet;
        send_offset[11] = (size_t)send_map.size_bdet;

        {
            MPI_Isend(send_offset.data(),12,SBD_MPI_SIZE_T,mpi_dest,id*4,comm,&req_size[0]);
        }
        {
            MPI_Irecv(recv_offset.data(),12,SBD_MPI_SIZE_T,mpi_source,id*4,comm,&req_size[1]);
        }
        {
            MPI_Waitall(2,req_size.data(),sta_size.data());
        }

        send_size = send_offset[0];
        recv_size = recv_offset[0];
        send_size_map = send_offset[1];
        recv_size_map = recv_offset[1];

        // Post async send/recv without any resize or fill kernel.
        // recv and recv_map_storage are pre-allocated to global_max by the caller.
        MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
        if( send_size != 0 ) {
            {
                MPI_Isend((ElemT*)thrust::raw_pointer_cast(send.data()),send_size,DataT,mpi_dest,id*4+2,comm,&req_send);
            }
        }
        if( recv_size != 0 ) {
            // No resize: recv is pre-allocated to global_max_ket_size by the caller.
            // MPI_Irecv writes recv_size elements directly into the device buffer.
            {
                MPI_Irecv((ElemT*)thrust::raw_pointer_cast(recv.data()),recv_size,DataT,mpi_source,id*4+2,comm,&req_recv);
            }
        }
        // recv_size == 0: nothing to receive; Sync() will return false for ket.

        if( send_size_map != 0 ) {
            {
                MPI_Isend((uint32_t*)thrust::raw_pointer_cast(send_map_storage.data()),send_size_map,MPI_UINT32_T,mpi_dest,id*4+3,comm,&req_send_map);
            }
        }

        // recv_map_base: pointer used to set up recv_map field offsets below.
        uint32_t* recv_map_base;
        if( recv_size_map != 0 ) {
            // No resize: recv_map_storage is pre-allocated to global_max_map_size.
            {
                MPI_Irecv((uint32_t*)thrust::raw_pointer_cast(recv_map_storage.data()),recv_size_map,MPI_UINT32_T,mpi_source,id*4+3,comm,&req_recv_map);
            }
            recv_map_base = thrust::raw_pointer_cast(recv_map_storage.data());
        } else {
            // recv_size_map == 0: degenerate case; point recv_map at send's map.
            // Sync() will not swap if recv_size is also 0.
            recv_offset = send_offset;
            recv_map_base = const_cast<uint32_t*>(thrust::raw_pointer_cast(send_map_storage.data()));
        }
        recv_map.AdetToDetOffset = recv_map_base + recv_offset[2];
        recv_map.BdetToDetOffset = recv_map_base + recv_offset[3];
        recv_map.AdetIndex = recv_map_base + recv_offset[4];
        recv_map.BdetIndex = recv_map_base + recv_offset[5];
        recv_map.AdetToBdetSM = recv_map_base + recv_offset[6];
        recv_map.AdetToDetSM = recv_map_base + recv_offset[7];
        recv_map.BdetToAdetSM = recv_map_base + recv_offset[8];
        recv_map.BdetToDetSM = recv_map_base + recv_offset[9];
        recv_map.size_adet = static_cast<uint32_t>(recv_offset[10]);
        recv_map.size_bdet = static_cast<uint32_t>(recv_offset[11]);
    }

    bool Sync(void)
    {
        bool recv = false;
        if (send_size > 0) {
            MPI_Status st;

            {
                MPI_Wait(&req_send, &st);
            }
        }
        if (recv_size > 0) {
            MPI_Status st;

            {
                MPI_Wait(&req_recv, &st);
            }
            recv = true;
        }
        if (send_size_map > 0) {
            MPI_Status st;

            {
                MPI_Wait(&req_send_map, &st);
            }
        }
        if (recv_size_map > 0) {
            MPI_Status st;

            {
                MPI_Wait(&req_recv_map, &st);
            }
            recv = true;
        }

        send_size = 0;
        recv_size = 0;
        send_size_map = 0;
        recv_size_map = 0;
        return recv;
    }
};




} // end namespace gdb

} // end namespace sbd

#endif
