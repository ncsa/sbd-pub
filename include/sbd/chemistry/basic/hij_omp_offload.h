/**
@file sbd/chemistry/basic/hij_omp_offload.h
@brief OpenMP target offload implementation of batched Hij computation
*/
#ifndef SBD_CHEMISTRY_BASIC_HIJ_OMP_OFFLOAD_H
#define SBD_CHEMISTRY_BASIC_HIJ_OMP_OFFLOAD_H

#ifdef USE_HIJ_OMP_OFFLOAD

// Compile-time debug flag: define HIJ_GPU_DEBUG to enable debug output
// This adds NO runtime overhead when not defined
#ifdef HIJ_GPU_DEBUG
#include <cstdio>
#define HIJ_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define HIJ_DEBUG_PRINT(...)                                                   \
  do {                                                                         \
  } while (0)
#endif

#include <limits> // For std::numeric_limits (bisection test)

namespace sbd {

// Declare all device functions for OpenMP target offload
#pragma omp declare target

// Device-side bit counting (portable popcount)
inline int popcount_device(size_t x) {
  int count = 0;
  while (x) {
    count += x & 1;
    x >>= 1;
  }
  return count;
}

// Device-side parity computation (5-parameter version)
template <typename ElemT>
inline void parity_device(const size_t *dets, size_t bit_length, int start,
                          int end, double &sgn) {
  if (start > end)
    return;

  size_t blockStart = start / bit_length;
  size_t bitStart = start % bit_length;
  size_t blockEnd = end / bit_length;
  size_t bitEnd = end % bit_length;

  int nonZeroBits = 0;

  if (blockStart == blockEnd) {
    size_t mask = ((size_t(1) << bitEnd) - 1) ^ ((size_t(1) << bitStart) - 1);
    nonZeroBits += popcount_device(dets[blockStart] & mask);
  } else {
    if (bitStart != 0) {
      size_t mask = ~((size_t(1) << bitStart) - 1);
      nonZeroBits += popcount_device(dets[blockStart] & mask);
      blockStart++;
    }

    for (size_t i = blockStart; i < blockEnd; i++) {
      nonZeroBits += popcount_device(dets[i]);
    }

    if (bitEnd != 0) {
      size_t mask = (size_t(1) << bitEnd) - 1;
      nonZeroBits += popcount_device(dets[blockEnd] & mask);
    }
  }

  sgn *= (-2.0 * (nonZeroBits % 2) + 1.0);

  if ((dets[start / bit_length] >> (start % bit_length)) & 1) {
    sgn *= -1.0;
  }
}

// Device-side helper to check bit occupation
inline bool getocc_device(const size_t *det, size_t bit_length, int x) {
  size_t index = x / bit_length;
  size_t bit_pos = x % bit_length;
  return (det[index] >> bit_pos) & 1;
}

// Device-side helper to count set bits in determinant
inline int bitcount_device(const size_t *det, size_t bit_length, size_t L) {
  int count = 0;
  size_t full_words = L / bit_length;
  size_t remaining_bits = L % bit_length;

  for (size_t i = 0; i < full_words; ++i) {
    count += popcount_device(det[i]);
  }

  if (remaining_bits > 0) {
    size_t mask = (size_t(1) << remaining_bits) - 1;
    count += popcount_device(det[full_words] & mask);
  }

  return count;
}

// Device-side two-electron integral lookup with spin checking
// Mimics twoInt::Value(i,j,k,l) from integrals.h
// Returns 0.0 if spin checks fail: ((i%2 == j%2) && (k%2 == l%2))
// Otherwise computes compact index and returns I2[index]
template <typename ElemT>
inline ElemT twoInt_device(int i, int j, int k, int l, const ElemT *I2) {
  // Spin check: i and j must have same spin, k and l must have same spin
  if ((i & 1) != (j & 1) || (k & 1) != (l & 1)) {
    return ElemT(0);
  }

  // Extract spatial orbital indices (divide by 2)
  int I = i >> 1;
  int J = j >> 1;
  int K = k >> 1;
  int L = l >> 1;

  // Compute compact indices using triangular storage
  // ij_index = max(I,J)*(max(I,J)+1)/2 + min(I,J)
  int max_IJ = (I > J) ? I : J;
  int min_IJ = (I < J) ? I : J;
  int ij = max_IJ * (max_IJ + 1) / 2 + min_IJ;

  int max_KL = (K > L) ? K : L;
  int min_KL = (K < L) ? K : L;
  int kl = max_KL * (max_KL + 1) / 2 + min_KL;

  // Final compact storage index
  int a = (ij > kl) ? ij : kl;
  int b = (ij < kl) ? ij : kl;
  int idx = a * (a + 1) / 2 + b;

  return I2[idx];
}

// Device-side ZeroExcite evaluation
template <typename ElemT>
inline ElemT ZeroExcite_device(const size_t *det, size_t bit_length, size_t L,
                               const ElemT &I0, const ElemT *I1,
                               const ElemT *I2_Direct, const ElemT *I2_Exchange,
                               int norbs_spatial, int norbs_spin) {
  ElemT energy = I0;

  HIJ_DEBUG_PRINT(
      "    ZeroExcite: L=%zu, norbs_spatial=%d, norbs_spin=%d, I0=%.6f\n", L,
      norbs_spatial, norbs_spin, double(I0));

  // Get closed orbitals
  int closed[256]; // Maximum 256 orbitals
  int num_closed = 0;

  for (int i = 0; i < 2 * L; i++) {
    if (getocc_device(det, bit_length, i)) {
      closed[num_closed++] = i;
    }
  }

  HIJ_DEBUG_PRINT("    ZeroExcite: num_closed=%d, closed orbitals:",
                  num_closed);
#ifdef HIJ_GPU_DEBUG
  for (int i = 0; i < num_closed; i++) {
    printf(" %d", closed[i]);
  }
  printf("\n");
#endif

  // One-electron contribution (use spin orbital indexing)
  for (int i = 0; i < num_closed; i++) {
    int I = closed[i];
    int idx = I * norbs_spin + I; //use norbs_spin for oneInt indexing
    ElemT h_val = I1[idx];
    HIJ_DEBUG_PRINT("    ZeroExcite: h[%d,%d] (idx=%d) = %.6f\n", I, I, idx,
                    double(h_val));
    energy += h_val;
  }

  // Two-electron contribution
  for (int i = 0; i < num_closed; i++) {
    int I = closed[i];
    for (int j = i + 1; j < num_closed; j++) {
      int J = closed[j];
      int orbital_I = I / 2;
      int orbital_J = J / 2;

      // Direct term (spin-independent) - use DirectMat accessor
      int idx = orbital_I + norbs_spatial * orbital_J;
      energy += I2_Direct[idx];

      // Exchange term (same spin only) - use ExchangeMat accessor
      if ((I % 2) == (J % 2)) {
        energy -= I2_Exchange[idx];
      }
    }
  }

  return energy;
}

// Device-side OneExcite evaluation
template <typename ElemT>
inline ElemT OneExcite_device(const size_t *det, size_t bit_length, int i,
                              int a, const ElemT *I1, const ElemT *I2,
                              int norbs_spatial, int norbs_spin) {
  double sgn = 1.0;
  parity_device<ElemT>(det, bit_length, (i < a) ? i : a, (i > a) ? i : a, sgn);

  HIJ_DEBUG_PRINT(
      "    OneExcite: i=%d, a=%d, norbs_spatial=%d, norbs_spin=%d, sgn=%.2f\n",
      i, a, norbs_spatial, norbs_spin, sgn);

  int idx_ai = a * norbs_spin + i; //use norbs_spin for oneInt indexing
  ElemT energy = I1[idx_ai];
  HIJ_DEBUG_PRINT("    OneExcite: h[%d,%d] (idx=%d) = %.6f\n", a, i, idx_ai,
                  double(energy));

  // Loop over occupied orbitals (use spin orbitals = 2 * norbs_spatial)
  // Combined loop using unconditional masking (no branch divergence)
  size_t num_words = (norbs_spin + bit_length - 1) / bit_length;
  size_t remaining_bits = norbs_spin % bit_length;

  // Precompute mask array: all bits set except last word may be partial
  size_t masks[16]; // Max 16 words for 1024 spin orbitals (512 spatial)
  for (size_t x = 0; x < num_words - 1; x++) {
    masks[x] = ~size_t(0); // All bits set
  }
  masks[num_words - 1] =
      (remaining_bits > 0) ? ((size_t(1) << remaining_bits) - 1) : ~size_t(0);

  for (size_t x = 0; x < num_words; x++) {
    size_t bits = det[x] & masks[x]; // Unconditional mask application
    int pos = 0;
    while (bits != 0) {
      if (bits & 1) {
        int j = x * bit_length + pos;

        // Use twoInt_device helper for cleaner code
        // I2.Value(a,i,j,j) - I2.Value(a,j,j,i)
        ElemT term1 = twoInt_device(a, i, j, j, I2);
        ElemT term2 = twoInt_device(a, j, j, i, I2);

        energy += term1 - term2;
      }
      bits >>= 1;
      pos++;
    }
  }

  return ElemT(sgn) * energy;
}

// Device-side TwoExcite evaluation
template <typename ElemT>
inline ElemT TwoExcite_device(const size_t *det, size_t bit_length, int i,
                              int j, int a, int b, const ElemT *I2) {
  double sgn = 1.0;
  int I = (i < j) ? i : j;
  int J = (i > j) ? i : j;
  int A = (a < b) ? a : b;
  int B = (a > b) ? a : b;

  parity_device<ElemT>(det, bit_length, (I < A) ? I : A, (I > A) ? I : A, sgn);
  parity_device<ElemT>(det, bit_length, (J < B) ? J : B, (J > B) ? J : B, sgn);

  if (A > J || B < I)
    sgn *= -1.0;

  // Use twoInt_device helper for proper spin checking
  // CPU code: I2.Value(A,I,B,J) - I2.Value(A,J,B,I)
  ElemT term1 = twoInt_device(A, I, B, J, I2);
  ElemT term2 = twoInt_device(A, J, B, I, I2);

  return ElemT(sgn) * (term1 - term2);
}

// Main device-side ComputeHij function
template <typename ElemT>
inline ElemT ComputeHij(const size_t *DetI, const size_t *DetJ,
                        size_t bit_length, size_t norbs, const ElemT &I0,
                        const ElemT *I1, const ElemT *I2,
                        const ElemT *I2_Direct, const ElemT *I2_Exchange) {
  int c[2] = {0, 0};
  int d[2] = {0, 0};
  int nc = 0;
  int nd = 0;

  size_t full_words = (2 * norbs) / bit_length;
  size_t remaining_bits = (2 * norbs) % bit_length;

  // Count orbital differences
  for (size_t i = 0; i < full_words; ++i) {
    size_t diff_c = DetI[i] & ~DetJ[i];
    size_t diff_d = DetJ[i] & ~DetI[i];

    for (size_t bit_pos = 0; bit_pos < bit_length; ++bit_pos) {
      if (diff_c & (size_t(1) << bit_pos)) {
        if (nc < 2)
          c[nc] = i * bit_length + bit_pos;
        nc++;
      }
      if (diff_d & (size_t(1) << bit_pos)) {
        if (nd < 2)
          d[nd] = i * bit_length + bit_pos;
        nd++;
      }
    }
  }

  if (remaining_bits > 0) {
    size_t mask = (size_t(1) << remaining_bits) - 1;
    size_t diff_c = (DetI[full_words] & ~DetJ[full_words]) & mask;
    size_t diff_d = (DetJ[full_words] & ~DetI[full_words]) & mask;

    for (size_t bit_pos = 0; bit_pos < remaining_bits; ++bit_pos) {
      if (diff_c & (size_t(1) << bit_pos)) {
        if (nc < 2)
          c[nc] = bit_length * full_words + bit_pos;
        nc++;
      }
      if (diff_d & (size_t(1) << bit_pos)) {
        if (nd < 2)
          d[nd] = bit_length * full_words + bit_pos;
        nd++;
      }
    }
  }

  // Dispatch based on excitation level
  HIJ_DEBUG_PRINT("ComputeHij: nc=%d, nd=%d\n", nc, nd);
  if (nc >= 1 && nc <= 2) {
    HIJ_DEBUG_PRINT("  c[0]=%d, d[0]=%d", c[0], d[0]);
    if (nc == 2) {
      HIJ_DEBUG_PRINT(", c[1]=%d, d[1]=%d", c[1], d[1]);
    }
    HIJ_DEBUG_PRINT("\n");
  }

  ElemT result = ElemT(0.0);
  if (nc == 0) {
    // ZeroExcite_device expects norbs_spatial as 3rd parameter (same as CPU
    // ZeroExcite) CPU: ZeroExcite(det, bit_length, L, I0, I1, I2) where L =
    // spatial orbitals
    result = ZeroExcite_device(DetJ, bit_length, norbs, I0, I1,
			       I2_Direct,   // Direct matrix
			       I2_Exchange, // Exchange matrix
			       norbs,       // spatial orbitals
			       2 * norbs);  // spins
    HIJ_DEBUG_PRINT("  ZeroExcite(norbs=%zu) -> %.6f\n", norbs, double(result));
  } else if (nc == 1) {
    result = OneExcite_device(DetJ, bit_length, d[0], c[0], I1, I2,
			      norbs,        // like above: spatial
                              2 * norbs     // spins
			      );
    HIJ_DEBUG_PRINT("  OneExcite(i=%d, a=%d) -> %.6f\n", d[0], c[0],
                    double(result));
  } else if (nc == 2) {
    result = TwoExcite_device(DetJ, bit_length, d[0], d[1], c[0], c[1], I2);
    HIJ_DEBUG_PRINT("  TwoExcite(i=%d, j=%d, a=%d, b=%d) -> %.6f\n", d[0], d[1],
                    c[0], c[1], double(result));
  } else {
    HIJ_DEBUG_PRINT("  nc=%d > 2, returning 0\n", nc);
  }

  return result;
}

// Device-side determinant computation (TRADMODE version)
inline void ComputeDetFromAlphaBeta_TRADMODE(const size_t *A, const size_t *B,
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
    for (int i = half; i < extra; i++) {
      if (A[j] & (1L << i))
        D[2 * j + 1] |= (1L << (2 * i - bit_length));
      if (B[j] & (1L << i))
        D[2 * j + 1] |= (1L << (2 * i + 1 - bit_length));
    }
  }
}

// Device-side determinant computation (Naive version)
inline void ComputeDetFromAlphaBeta_Naive(const size_t *A, const size_t *B,
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

// Batched Hij computation on GPU using OpenMP target offload
template <typename ElemT>
void Hij_Batch_OMP(const size_t *det_cache, const size_t *bra_ia,
                   const size_t *bra_ib, const size_t *ket_ja,
                   const size_t *ket_jb, const ElemT &I0, const ElemT *I1,
                   const ElemT *I2, ElemT *hij_results, size_t batch_size,
                   size_t bit_length, size_t norbs, size_t n_alpha,
                   size_t n_beta, size_t det_size, size_t I1_size,
                   size_t I2_size) {

  // det_cache is already GPU-resident (target enter data)
  // Use map(alloc:) to reuse existing GPU allocation without copying
  size_t det_cache_size = n_alpha * n_beta * det_size;

#pragma omp target teams distribute parallel for map(                          \
        to : bra_ia[0 : batch_size], bra_ib[0 : batch_size],                   \
            ket_ja[0 : batch_size], ket_jb[0 : batch_size], I1[0 : I1_size],   \
            I2[0 : I2_size], I0, n_beta, det_size, bit_length, norbs)          \
    map(alloc : det_cache[0 : det_cache_size])                                 \
    map(from : hij_results[0 : batch_size])
  for (size_t idx = 0; idx < batch_size; idx++) {
    size_t bra_offset = (bra_ia[idx] * n_beta + bra_ib[idx]) * det_size;
    size_t ket_offset = (ket_ja[idx] * n_beta + ket_jb[idx]) * det_size;

    const size_t *DetI = &det_cache[bra_offset];
    const size_t *DetJ = &det_cache[ket_offset];

    hij_results[idx] = ComputeHij(DetI, DetJ, bit_length, norbs, I0, I1, I2);
  }
}

} // namespace sbd

#endif // USE_HIJ_OMP_OFFLOAD

#endif // SBD_CHEMISTRY_BASIC_HIJ_OMP_OFFLOAD_H
