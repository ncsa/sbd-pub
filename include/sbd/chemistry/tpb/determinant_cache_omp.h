/**
@file sbd/chemistry/tpb/determinant_cache_omp.h
@brief OpenMP target offload implementation of determinant precomputation cache
*/
#ifndef SBD_CHEMISTRY_TPB_DETERMINANT_CACHE_OMP_H
#define SBD_CHEMISTRY_TPB_DETERMINANT_CACHE_OMP_H

#include <cstring>
#include <mpi.h>
#include <vector>

namespace sbd {

/**
 * @brief Precomputes all (n_alpha × n_beta) determinants using OpenMP target
 * offload
 *
 * This class precomputes all full determinants from alpha and beta determinant
 * pairs and stores them in a cache on the GPU device. The cache eliminates
 * redundant DetFromAlphaBeta calls in mult() inner loops.
 *
 * @tparam ElemT Element type (typically double)
 */
template <typename ElemT> class DeterminantCacheOMP {
public:
  /**
   * @brief Constructor: precomputes determinants and transfers to GPU
   *
   * @param adets Alpha determinants (vector of bitstings)
   * @param bdets Beta determinants (vector of bitstrings)
   * @param bit_length Size of size_t in bits (typically 64)
   * @param norbs Number of orbitals
   * @param h_comm MPI communicator for task distribution
   */
  DeterminantCacheOMP(const std::vector<std::vector<size_t>> &adets,
                      const std::vector<std::vector<size_t>> &bdets,
                      size_t bit_length, size_t norbs, MPI_Comm h_comm)
      : bit_length_(bit_length), norbs_(norbs) {

    n_alpha_ = adets.size();
    n_beta_ = bdets.size();

    // Calculate size of each determinant
    det_size_per_det_ = (2 * norbs + bit_length - 1) / bit_length;
    size_per_alpha_det_ = (norbs + bit_length - 1) / bit_length;
    size_per_beta_det_ = (norbs + bit_length - 1) / bit_length;

    // Allocate cache for all determinants
    size_t total_dets = n_alpha_ * n_beta_;
    det_cache_.resize(total_dets * det_size_per_det_, 0);

    // Flatten alpha and beta determinants for GPU transfer
    std::vector<size_t> alpha_flat(n_alpha_ * size_per_alpha_det_);
    std::vector<size_t> beta_flat(n_beta_ * size_per_beta_det_);

    for (size_t ia = 0; ia < n_alpha_; ia++) {
      for (size_t j = 0; j < adets[ia].size(); j++) {
        alpha_flat[ia * size_per_alpha_det_ + j] = adets[ia][j];
      }
    }

    for (size_t ib = 0; ib < n_beta_; ib++) {
      for (size_t j = 0; j < bdets[ib].size(); j++) {
        beta_flat[ib * size_per_beta_det_ + j] = bdets[ib][j];
      }
    }

    // Get raw pointers for OpenMP offload
    size_t *det_cache_ptr = det_cache_.data();
    const size_t *alpha_ptr = alpha_flat.data();
    const size_t *beta_ptr = beta_flat.data();

    size_t n_alpha = n_alpha_;
    size_t n_beta = n_beta_;
    size_t det_size = det_size_per_det_;
    size_t alpha_size = size_per_alpha_det_;
    size_t beta_size = size_per_beta_det_;
    size_t cache_size = det_cache_.size();
    size_t alpha_flat_size = alpha_flat.size();
    size_t beta_flat_size = beta_flat.size();

// Allocate cache on GPU and keep it resident (do NOT use map(from:))
// Use target enter data to create persistent GPU allocation
#pragma omp target enter data map(alloc : det_cache_ptr[0 : cache_size])

// Precompute determinants on GPU using OpenMP target offload
// Note: No map() clause for det_cache_ptr - already on GPU from enter data
#pragma omp target teams distribute parallel for collapse(2)                   \
    map(to : alpha_ptr[0 : alpha_flat_size], beta_ptr[0 : beta_flat_size],     \
            bit_length, norbs, n_alpha, n_beta, det_size, alpha_size,          \
            beta_size)
    for (size_t ia = 0; ia < n_alpha; ia++) {
      for (size_t ib = 0; ib < n_beta; ib++) {
        size_t offset = (ia * n_beta + ib) * det_size;

// Call DetFromAlphaBeta inline
// We use the TRADMODE version for best performance
#ifdef SBD_TRADMODE
        ComputeDetFromAlphaBeta_TRADMODE(
            &alpha_ptr[ia * alpha_size], &beta_ptr[ib * beta_size],
            &det_cache_ptr[offset], bit_length, norbs, det_size);
#else
        ComputeDetFromAlphaBeta_Naive(
            &alpha_ptr[ia * alpha_size], &beta_ptr[ib * beta_size],
            &det_cache_ptr[offset], bit_length, norbs, det_size);
#endif
      }
    }

    // Note: GPU memory stays allocated - cache remains GPU-resident
    // Will be freed in destructor with target exit data
  }

  /**
   * @brief Destructor: free GPU-resident cache
   */
  ~DeterminantCacheOMP() {
    size_t *det_cache_ptr = det_cache_.data();
    size_t cache_size = det_cache_.size();
#pragma omp target exit data map(delete : det_cache_ptr[0 : cache_size])
  }

  /**
   * @brief Get a precomputed determinant from cache (downloads from GPU)
   *
   * @param ia Alpha determinant index
   * @param ib Beta determinant index
   * @return Vector containing the full determinant
   */
  std::vector<size_t> GetDeterminant(size_t ia, size_t ib) const {
    // Need to download from GPU to host
    size_t offset = (ia * n_beta_ + ib) * det_size_per_det_;
    std::vector<size_t> result(det_size_per_det_);

    size_t *det_cache_ptr = const_cast<size_t *>(det_cache_.data());
    size_t *result_ptr = result.data();
    size_t count = det_size_per_det_;

// Transfer single determinant from GPU to host
#pragma omp target update from(det_cache_ptr[offset : count])
    std::memcpy(result_ptr, &det_cache_ptr[offset], count * sizeof(size_t));

    return result;
  }

  /**
   * @brief Get determinant into pre-allocated buffer (downloads from GPU)
   *
   * @param ia Alpha determinant index
   * @param ib Beta determinant index
   * @param det Output buffer (must be size det_size_per_det_)
   */
  void GetDeterminant(size_t ia, size_t ib, std::vector<size_t> &det) const {
    size_t offset = (ia * n_beta_ + ib) * det_size_per_det_;
    size_t *det_cache_ptr = const_cast<size_t *>(det_cache_.data());
    size_t count = det_size_per_det_;

// Transfer single determinant from GPU to host
#pragma omp target update from(det_cache_ptr[offset : count])
    std::memcpy(det.data(), &det_cache_ptr[offset], count * sizeof(size_t));
  }

  /**
   * @brief Get the size of each determinant in size_t elements
   *
   * @return Size of each full determinant
   */
  size_t GetDeterminantSize() const { return det_size_per_det_; }

  /**
   * @brief Get raw pointer to cache for GPU kernel access
   *
   * @return Pointer to beginning of determinant cache
   */
  const size_t *GetCachePointer() const { return det_cache_.data(); }

private:
  std::vector<size_t> det_cache_; ///< Flattened determinant cache
  size_t n_alpha_;                ///< Number of alpha determinants
  size_t n_beta_;                 ///< Number of beta determinants
  size_t det_size_per_det_;       ///< Size of each full determinant
  size_t size_per_alpha_det_;     ///< Size of each alpha determinant
  size_t size_per_beta_det_;      ///< Size of each beta determinant
  size_t bit_length_;             ///< Bits per size_t
  size_t norbs_;                  ///< Number of orbitals

/**
 * @brief TRADMODE version of DetFromAlphaBeta (optimized bit interleaving)
 *
 * This is the inline GPU version of the TRADMODE DetFromAlphaBeta.
 * Must be declared pragma omp declare target for GPU execution.
 */
#pragma omp declare target
  static void ComputeDetFromAlphaBeta_TRADMODE(const size_t *A, const size_t *B,
                                               size_t *D, size_t bit_length,
                                               size_t L, size_t dsize) {
    int fsize = L / bit_length;
    int half = bit_length / 2;
    int extra = L - fsize * bit_length;
    int case1 = (extra > 0) && (extra <= half);
    int case2 = (extra > 0) && (extra > half);

    for (int j = 0; j < fsize; j++) {
      D[2 * j] = 0;
      D[2 * j + 1] = 0;
      for (int i = 0; i < half; i++) {
        if (A[j] & (1L << i))
          D[2 * j] |= (1L << (2 * i));
        if (B[j] & (1L << i))
          D[2 * j] |= (1L << (2 * i + 1));
        if (A[j] & (1L << (i + half)))
          D[2 * j + 1] |= (1L << (2 * i));
        if (B[j] & (1L << (i + half)))
          D[2 * j + 1] |= (1L << (2 * i + 1));
      }
    }

    if (case1) {
      int j = fsize;
      D[2 * j] = 0;
      for (int i = 0; i < extra; i++) {
        if (A[j] & (1L << i))
          D[2 * j] |= (1L << (2 * i));
        if (B[j] & (1L << i))
          D[2 * j] |= (1L << (2 * i + 1));
      }
    }

    if (case2) {
      int j = fsize;
      D[2 * j] = 0;
      D[2 * j + 1] = 0;
      for (int i = 0; i < half; i++) {
        if (A[j] & (1L << i))
          D[2 * j] |= (1L << (2 * i));
        if (B[j] & (1L << i))
          D[2 * j] |= (1L << (2 * i + 1));
      }
      for (int i = 0; i < extra - half; i++) {
        if (A[j] & (1L << (i + half)))
          D[2 * j + 1] |= (1L << (2 * i));
        if (B[j] & (1L << (i + half)))
          D[2 * j + 1] |= (1L << (2 * i + 1));
      }
    }
  }

  /**
   * @brief Naive version of DetFromAlphaBeta (for non-TRADMODE builds)
   */
  static void ComputeDetFromAlphaBeta_Naive(const size_t *A, const size_t *B,
                                            size_t *D, size_t bit_length,
                                            size_t L, size_t dsize) {
    // Initialize D to zero
    for (size_t i = 0; i < dsize; i++) {
      D[i] = 0;
    }

    for (size_t i = 0; i < L; i++) {
      size_t block = i / bit_length;
      size_t bit_pos = i % bit_length;
      size_t new_block_A = (2 * i) / bit_length;
      size_t new_bit_pos_A = (2 * i) % bit_length;
      size_t new_block_B = (2 * i + 1) / bit_length;
      size_t new_bit_pos_B = (2 * i + 1) % bit_length;

      if (A[block] & (size_t(1) << bit_pos)) {
        D[new_block_A] |= size_t(1) << new_bit_pos_A;
      }
      if (B[block] & (size_t(1) << bit_pos)) {
        D[new_block_B] |= size_t(1) << new_bit_pos_B;
      }
    }
  }
#pragma omp end declare target
};

} // namespace sbd

#endif // SBD_CHEMISTRY_TPB_DETERMINANT_CACHE_OMP_H
