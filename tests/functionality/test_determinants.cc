/**
 * @file test_determinants.cc
 * @brief Unit and regression tests for DetFromAlphaBeta.
 */

#include "sbd/sbd.h"
#include "utils.h"
#include <mpi.h>

using namespace sbd;

//=============================================================================
// DetFromAlphaBeta Unit Tests
//=============================================================================

void test_DetFromAlphaBeta_simple() {
  const size_t bit_length = 64;
  const size_t L = 4; // 4 orbitals

  std::vector<size_t> A = {0b0011}; // Alpha: |0011⟩
  std::vector<size_t> B = {0b0101}; // Beta:  |0101⟩

  auto D1 = DetFromAlphaBeta(A, B, bit_length, L);
  std::vector<size_t> D2((2 * L + bit_length - 1) / bit_length, 0);
  DetFromAlphaBeta(A, B, bit_length, L, D2);

  TEST_ASSERT(D1.size() == D2.size());
  for (size_t i = 0; i < D1.size(); i++) {
    TEST_ASSERT(D1[i] == D2[i]);
  }

  TEST_ASSERT((D1[0] & 0b11) == 0b11);
  TEST_ASSERT((D1[0] & 0b1100) == 0b0100);
  TEST_ASSERT((D1[0] & 0b110000) == 0b100000);
  TEST_ASSERT((D1[0] & 0b11000000) == 0);
}

void test_DetFromAlphaBeta_multiword() {
  const size_t bit_length = 64;
  const size_t L = 100;

  std::vector<size_t> A((L + bit_length - 1) / bit_length, 0);
  for (size_t i = 0; i < L; i += 5) {
    A[i / bit_length] |= (size_t(1) << (i % bit_length));
  }

  std::vector<size_t> B((L + bit_length - 1) / bit_length, 0);
  for (size_t i = 0; i < L; i += 7) {
    B[i / bit_length] |= (size_t(1) << (i % bit_length));
  }

  auto D1 = DetFromAlphaBeta(A, B, bit_length, L);

  for (size_t orb = 0; orb < L; orb++) {
    size_t alpha_bit = 2 * orb;
    size_t beta_bit = 2 * orb + 1;
    bool alpha_set =
        (D1[alpha_bit / bit_length] >> (alpha_bit % bit_length)) & 1;
    bool beta_set = (D1[beta_bit / bit_length] >> (beta_bit % bit_length)) & 1;
    TEST_ASSERT(alpha_set == (orb % 5 == 0));
    TEST_ASSERT(beta_set == (orb % 7 == 0));
  }
}

void test_DetFromAlphaBeta_edge_cases() {
  const size_t bit_length = 64;
  size_t L = 4;

  // Empty
  TEST_ASSERT(DetFromAlphaBeta({0}, {0}, bit_length, L)[0] == 0);
  // Full
  TEST_ASSERT(DetFromAlphaBeta({0b1111}, {0b1111}, bit_length, L)[0] ==
              0b11111111);
  // Alpha only
  TEST_ASSERT(DetFromAlphaBeta({0b1010}, {0}, bit_length, L)[0] == 0b01000100);
  // Beta only
  TEST_ASSERT(DetFromAlphaBeta({0}, {0b1010}, bit_length, L)[0] == 0b10001000);
}

void test_DetFromAlphaBeta_regression_Fe4S4() {
  const size_t bit_length = 64;
  const size_t L = 36;
  std::vector<size_t> A = {0b111111111111111111111111}; // 24 electrons
  std::vector<size_t> B = {0b11111111111111111111};     // 20 electrons

  auto D1 = DetFromAlphaBeta(A, B, bit_length, L);

  int count = 0;
  for (size_t i = 0; i < D1.size(); i++) {
    count += __builtin_popcountll(D1[i]);
  }
  TEST_ASSERT(count == 24 + 20);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  std::vector<TestResult> results;

  results.push_back(
      run_test("DetFromAlphaBeta - simple", test_DetFromAlphaBeta_simple));
  results.push_back(run_test("DetFromAlphaBeta - multiword",
                             test_DetFromAlphaBeta_multiword));
  results.push_back(run_test("DetFromAlphaBeta - edge cases",
                             test_DetFromAlphaBeta_edge_cases));
  results.push_back(run_test("DetFromAlphaBeta - Fe4S4 regression",
                             test_DetFromAlphaBeta_regression_Fe4S4));

  int failed = 0;
  for (const auto &result : results) {
    if (!result.passed) {
      failed++;
      std::cout << "✗ FAIL: " << result.name << " (" << result.error_msg
                << ")\n";
    } else {
      std::cout << "✓ PASS: " << result.name << "\n";
    }
  }

  MPI_Finalize();
  return (failed == 0) ? 0 : 1;
}
