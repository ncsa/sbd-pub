/**
@file sbd/chemistry/basic/determinants_thrust.h
@brief Functions to handle the bit-string basis for Thrust
*/

#ifndef SBD_CHEMISTRY_BASIC_DETERMINANTS_THRUST_H
#define SBD_CHEMISTRY_BASIC_DETERMINANTS_THRUST_H

namespace sbd
{

template <typename ElemT>
class DeterminantKernels {
protected:
    ElemT I0;
    oneInt_Thrust<ElemT> I1;
    twoInt_Thrust<ElemT> I2;
    uint32_t bit_length;
    size_t norbs;
    size_t D_size;      // the vector length of a full (i.e., alpha + beta) determinant
    size_t D_half_size; // the vector length of a half (i.e., alpha or beta) determinant
public:
    DeterminantKernels() {}

    DeterminantKernels(const uint32_t bit_length_in, const size_t norbs_in,
                        const ElemT zero_in,
                        const oneInt_Thrust<ElemT> one_in,
                        const twoInt_Thrust<ElemT> two_in
                ) : I0(zero_in), I1(one_in), I2(two_in)
    {
        bit_length = bit_length_in;
        norbs = norbs_in;
        D_size = (2 * norbs + bit_length - 1) / bit_length;
        D_half_size = (norbs + bit_length - 1) / bit_length;
    }

    __device__ __host__ void DetFromAlphaBeta(size_t *D, const size_t *A, const size_t *B)
    {
        size_t i;
        for (i = 0; i < D_size; i++) {
            D[i] = 0;
        }
        for (i = 0; i < norbs; i++) {
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

    inline __device__ __host__ void parity(const size_t* dets, const int start, const int end, double& sgn)
    {
        const int blockStart = start / bit_length;
        const int bitStart = start - (blockStart * bit_length);

        const int blockEnd = end / bit_length;
        const int bitEnd = end - (blockEnd * bit_length);

        int nonZeroBits = 0; // counter for nonzero bits

        // Preserve the original parity convention, where the bit at 'start'
        // contributes through an extra sign rule in addition to the range count.
        // We fold that contribution into nonZeroBits here to avoid an extra branch.
        nonZeroBits = (dets[blockStart] >> bitStart) & 1;

        // 1. Count bits in the start block
        if (blockStart == blockEnd) {
            // the case where start and end is same block
            // size_t mask = ((size_t(1) << bitEnd) - 1) ^ ((size_t(1) << bitStart) - 1);
            size_t mask = ((size_t(1) << (bitEnd - bitStart)) - 1) << bitStart;
            nonZeroBits += __popcll(dets[blockStart] & mask);
        }
        else {
            int i = blockStart;
            // 2. Handle the partial bits in the start block
            if (bitStart != 0) {
                size_t mask = ~((size_t(1) << bitStart) - 1); // count after bitStart
                nonZeroBits += __popcll(dets[blockStart] & mask);
                i++;
            }

            // 3. Handle full blocks in between
            for (; i < blockEnd; i++) {
                nonZeroBits += __popcll(dets[i]);
            }

            // 4. Handle the partial bits in the end block
            if (bitEnd != 0) {
                size_t mask = (size_t(1) << bitEnd) - 1; // count before bitEnd
                nonZeroBits += __popcll(dets[blockEnd] & mask);
            }
        }

        // parity estimation
        sgn *= (-2. * (nonZeroBits & 1) + 1.);

        // // flip sign if start == 1
        // if ((dets[blockStart] >> bitStart) & 1) {
        //     sgn *= -1.;
        // }
    }

#ifdef SBD_USE_32BIT_PARITY
    //
    // 32-bit parity implementation (register-optimized path).
    //
    // Use 32-bit popcount (__popc) instead of 64-bit (__popcll) to reduce
    // register pressure in GPU kernels. Lower register usage can improve
    // occupancy and overall performance.
    //
    // The det array is stored as size_t (64-bit), but only the lower
    // bit_length bits are used. By casting to uint32_t*, we operate on
    // the lower 32 bits of each element.
    //
    // Constraint:
    //   bit_length <= 32
    //
    inline __device__ __host__ void parity(const uint32_t* dets, const int start, const int end, double& sgn)
    {
        const int blockStart = start / bit_length;
        const int bitStart = start - (blockStart * bit_length);

        const int blockEnd = end / bit_length;
        const int bitEnd = end - (blockEnd * bit_length);

        int nonZeroBits = 0; // counter for nonzero bits

        // Preserve the original parity convention, where the bit at 'start'
        // contributes through an extra sign rule in addition to the range count.
        // We fold that contribution into nonZeroBits here to avoid an extra branch.
        nonZeroBits = (dets[blockStart*2] >> bitStart) & 1;

        // 1. Count bits in the start block
        if (blockStart == blockEnd) {
            // the case where start and end is same block
            // uint32_t mask = ((uint32_t(1) << bitEnd) - 1) ^ ((uint32_t(1) << bitStart) - 1);
            uint32_t mask = ((uint32_t(1) << (bitEnd - bitStart)) - 1) << bitStart;
            nonZeroBits += __popc(dets[blockStart*2] & mask);
        }
        else {
            int i = blockStart;
            // 2. Handle the partial bits in the start block
            if (bitStart != 0) {
                uint32_t mask = ~((uint32_t(1) << bitStart) - 1); // count after bitStart
                nonZeroBits += __popc(dets[blockStart*2] & mask);
                i++;
            }

            // 3. Handle full blocks in between
            for (; i < blockEnd; i++) {
                nonZeroBits += __popc(dets[i*2]);
            }

            // 4. Handle the partial bits in the end block
            if (bitEnd != 0) {
                uint32_t mask = (uint32_t(1) << bitEnd) - 1; // count before bitEnd
                nonZeroBits += __popc(dets[blockEnd*2] & mask);
            }
        }

        // parity estimation
        sgn *= (-2. * (nonZeroBits & 1) + 1.);

        // // flip sign if start == 1
        // if ((dets[blockStart*2] >> bitStart) & 1) {
        //     sgn *= -1.;
        // }
    }
#endif

    inline __device__ __host__ bool getocc(const size_t* det, int x)
    {
        int index = x / bit_length;
        int bit_pos = x % bit_length;
        return (det[index] >> bit_pos) & 1;
    }

    inline __device__ __host__ ElemT ZeroExcite(const size_t* det)
    {
        ElemT energy(0.0);

        for (int i = 0; i < 2 * norbs; i++) {
            if (getocc(det, i)) {
                energy += I1.Value(i, i);
                for (int j = i + 1; j < 2 * norbs; j++) {
                    if (getocc(det, j)) {
                        energy += I2.DirectValue(i, j);
                        if ((i % 2) == (j % 2)) {
                            energy -= I2.ExchangeValue(i, j);
                        }
                    }
                }
            }
        }
        return energy + I0;
    }

    inline __device__ __host__ ElemT OneExcite(const size_t* det, int i, int a)
    {
        double sgn = 1.0;
#ifdef SBD_USE_32BIT_PARITY
        parity((const uint32_t*)det, std::min(i, a), std::max(i, a), sgn);
#else
        parity(det, std::min(i, a), std::max(i, a), sgn);
#endif
        ElemT energy = I1.Value(a, i);
        for (int x = 0; x < D_size; x++) {
            size_t bits = det[x];
            for (int pos = 0; pos < bit_length; pos++) {
                if ((bits & 1ULL) == 1ULL) {
                    int j = x * bit_length + pos;
                    energy += (I2.Value(a, i, j, j) - I2.Value(a, j, j, i));
                }
                bits >>= 1;
            }
        }
        energy *= ElemT(sgn);
        return energy;
    }

    inline __device__ __host__ ElemT TwoExcite(const size_t* det, int i, int j, int a, int b)
    {
        double sgn = 1.0;
        int I = std::min(i, j);
        int J = std::max(i, j);
        int A = std::min(a, b);
        int B = std::max(a, b);
#ifdef SBD_USE_32BIT_PARITY
        parity((const uint32_t*)det, std::min(I, A), std::max(I, A), sgn);
        parity((const uint32_t*)det, std::min(J, B), std::max(J, B), sgn);
#else
        parity(det, std::min(I, A), std::max(I, A), sgn);
        parity(det, std::min(J, B), std::max(J, B), sgn);
#endif
        if (A > J || B < I) {
            sgn *= -1.0;
        }
        return ElemT(sgn) * (I2.Value(A, I, B, J) - I2.Value(A, J, B, I));
    }
};

}

#endif
