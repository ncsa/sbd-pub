/**
 * @file test_fe4s4_minimal.cc
 * @brief Minimal Fe4S4 test: GPU vs CPU Hij comparison with REAL data
 *
 * This test loads real Fe4S4 FCIDUMP and determinants, then compares
 * GPU ComputeHij against CPU Hij() for actual determinant pairs.
 *
 * Purpose: Catch bugs that synthetic tests miss due to:
 * - Multi-word determinants (Fe4S4 has 36 orbitals = 2 words)
 * - Real integral values from FCIDUMP
 * - Complex excitation patterns
 */

#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// GPU offload macros must be defined by Makefile with GPU=1
// Do NOT hardcode these - let Makefile control them

#include "sbd/sbd.h"

using namespace sbd;

const double REL_TOL = 1e-10;
const double ABS_TOL = 1e-12;

bool almost_equal(double a, double b, double rel_tol = REL_TOL,
                  double abs_tol = ABS_TOL) {
  if (std::abs(a - b) < abs_tol)
    return true;
  double max_val = std::max(std::abs(a), std::abs(b));
  if (max_val < abs_tol)
    return true;
  return std::abs(a - b) / max_val < rel_tol;
}

int main(int argc, char **argv) {
#ifdef USE_GPU
  std::cout << "========================================" << std::endl;
  std::cout << "Fe4S4 Minimal Correctness Test" << std::endl;
  std::cout << "GPU ComputeHij vs CPU Hij with REAL data" << std::endl;
  std::cout << "========================================" << std::endl;

  // Load REAL Fe4S4 FCIDUMP
  std::string fcidump_path =
      "../../samples/selected_basis_diagonalization/fcidump_Fe4S4.txt";

  std::cout << "\nLoading FCIDUMP: " << fcidump_path << std::endl;
  FCIDump fcidump = LoadFCIDump(fcidump_path);

  int L, N; // L = spatial orbitals, N = electrons
  double I0_val;
  oneInt<double> I1_cpu;
  twoInt<double> I2_cpu;

  SetupIntegrals(fcidump, L, N, I0_val, I1_cpu, I2_cpu);

  size_t norbs_spatial = L;
  std::cout << "  Spatial orbitals: " << norbs_spatial << std::endl;
  std::cout << "  Spin orbitals: " << 2 * norbs_spatial << std::endl;
  std::cout << "  Electrons: " << N << std::endl;
  std::cout << "  Nuclear repulsion: " << I0_val << std::endl;

  std::string adet_path =
      "../../samples/selected_basis_diagonalization/AlphaDets.txt";

  std::cout << "\nLoading alpha determinants: " << adet_path << std::endl;
  std::vector<std::vector<size_t>> adets;
  size_t bit_length = 64;

  LoadAlphaDets(adet_path, adets, bit_length, norbs_spatial);
  std::cout << "  Loaded " << adets.size() << " alpha determinants"
            << std::endl;

  auto bdets = adets;

  size_t I1_size = (2 * norbs_spatial) * (2 * norbs_spatial);
  size_t I2_size = I2_cpu.store.size();
  size_t I2_Direct_size = norbs_spatial * norbs_spatial;
  size_t I2_Exchange_size = norbs_spatial * norbs_spatial;

  std::vector<double> I1_flat(I1_size);
  std::vector<double> I2_flat(I2_size);
  std::vector<double> I2_Direct_flat(I2_Direct_size);
  std::vector<double> I2_Exchange_flat(I2_Exchange_size);

  std::cout << "\nFlattening integrals for GPU:" << std::endl;
  std::cout << "  I1 size: " << I1_size << " elements" << std::endl;
  std::cout << "  I2 size: " << I2_size << " elements" << std::endl;
  std::cout << "  I2_Direct size: " << I2_Direct_size << " elements"
            << std::endl;
  std::cout << "  I2_Exchange size: " << I2_Exchange_size << " elements"
            << std::endl;

  for (size_t i = 0; i < 2 * norbs_spatial; i++) {
    for (size_t j = 0; j < 2 * norbs_spatial; j++) {
      I1_flat[i * (2 * norbs_spatial) + j] = I1_cpu.Value(i, j);
    }
  }

  for (size_t ij = 0; ij < I2_size; ij++) {
    I2_flat[ij] = I2_cpu.store[ij];
  }

  for (size_t i = 0; i < norbs_spatial; i++) {
    for (size_t j = 0; j < norbs_spatial; j++) {
      I2_Direct_flat[i + norbs_spatial * j] = I2_cpu.DirectValue(i, j);
      I2_Exchange_flat[i + norbs_spatial * j] = I2_cpu.ExchangeValue(i, j);
    }
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "Testing ComputeHij vs Hij on real Fe4S4 pairs" << std::endl;
  std::cout << "========================================" << std::endl;

  struct TestPair {
    size_t ia_bra, ib_bra, ia_ket, ib_ket;
    std::string description;
  };

  std::vector<TestPair> test_pairs = {
      {0, 0, 0, 0, "Diagonal: Det[0,0]"},
      {1, 1, 1, 1, "Diagonal: Det[1,1]"},
      {5, 5, 5, 5, "Diagonal: Det[5,5]"},
      {10, 10, 10, 10, "Diagonal: Det[10,10]"},
      {20, 30, 20, 30, "Diagonal: Det[20,30]"},

      {0, 0, 1, 0, "Single alpha: (0,0) -> (1,0)"},
      {0, 0, 2, 0, "Single alpha: (0,0) -> (2,0)"},
      {5, 10, 6, 10, "Single alpha: (5,10) -> (6,10)"},
      {10, 20, 11, 20, "Single alpha: (10,20) -> (11,20)"},

      {0, 0, 0, 1, "Single beta: (0,0) -> (0,1)"},
      {0, 0, 0, 2, "Single beta: (0,0) -> (0,2)"},
      {5, 10, 5, 11, "Single beta: (5,10) -> (5,11)"},
      {10, 20, 10, 21, "Single beta: (10,20) -> (10,21)"},

      {0, 0, 2, 0, "Double alpha: (0,0) -> (2,0)"},
      {0, 0, 3, 0, "Double alpha: (0,0) -> (3,0)"},
      {5, 10, 7, 10, "Double alpha: (5,10) -> (7,10)"},
      {10, 20, 12, 20, "Double alpha: (10,20) -> (12,20)"},

      {0, 0, 0, 2, "Double beta: (0,0) -> (0,2)"},
      {0, 0, 0, 3, "Double beta: (0,0) -> (0,3)"},
      {5, 10, 5, 12, "Double beta: (5,10) -> (5,12)"},
      {10, 20, 10, 22, "Double beta: (10,20) -> (10,22)"},

      {0, 0, 1, 1, "Mixed double: (0,0) -> (1,1)"},
      {0, 0, 2, 2, "Mixed double: (0,0) -> (2,2)"},
      {5, 10, 6, 11, "Mixed double: (5,10) -> (6,11)"},
      {10, 20, 11, 21, "Mixed double: (10,20) -> (11,21)"},

      {0, 0, 10, 10, "Higher order: (0,0) -> (10,10)"},
      {0, 0, 20, 20, "Higher order: (0,0) -> (20,20)"},
      {100, 50, 120, 80, "Higher order: (100,50) -> (120,80)"},
  };

  int passed = 0;
  int failed = 0;
  int diagonal_passed = 0, diagonal_failed = 0;
  int single_passed = 0, single_failed = 0;
  int double_passed = 0, double_failed = 0;
  int higher_passed = 0, higher_failed = 0;

  for (const auto &test : test_pairs) {
    if (test.ia_bra >= adets.size() || test.ib_bra >= bdets.size() ||
        test.ia_ket >= adets.size() || test.ib_ket >= bdets.size()) {
      std::cout << "\nSkipping: " << test.description
                << " (indices out of range)" << std::endl;
      continue;
    }

    std::cout << "\n--- Test: " << test.description << " ---" << std::endl;
    std::cout << "  Bra: (α=" << test.ia_bra << ", β=" << test.ib_bra << ")"
              << std::endl;
    std::cout << "  Ket: (α=" << test.ia_ket << ", β=" << test.ib_ket << ")"
              << std::endl;

    // Build full determinants
    auto DetI = DetFromAlphaBeta(adets[test.ia_bra], bdets[test.ib_bra],
                                 bit_length, norbs_spatial);
    auto DetJ = DetFromAlphaBeta(adets[test.ia_ket], bdets[test.ib_ket],
                                 bit_length, norbs_spatial);

    // CPU Hij
    std::vector<int> c(2, 0);
    std::vector<int> d(2, 0);
    size_t orbDiff;
    double hij_cpu = Hij(DetI, DetJ, bit_length, norbs_spatial, c, d, I0_val,
                         I1_cpu, I2_cpu, orbDiff);

    // GPU ComputeHij
    double hij_gpu =
        ComputeHij(DetI.data(), DetJ.data(), bit_length, norbs_spatial, I0_val,
                   I1_flat.data(), I2_flat.data(), I2_Direct_flat.data(),
                   I2_Exchange_flat.data());

    std::cout << "  CPU Hij  = " << std::scientific << std::setprecision(16)
              << hij_cpu << std::endl;
    std::cout << "  GPU Hij  = " << std::scientific << std::setprecision(16)
              << hij_gpu << std::endl;

    bool is_diagonal =
        (test.ia_bra == test.ia_ket && test.ib_bra == test.ib_ket);
    bool is_single =
        (test.ia_bra == test.ia_ket && test.ib_bra != test.ib_ket) ||
        (test.ia_bra != test.ia_ket && test.ib_bra == test.ib_ket);
    bool is_double = !is_diagonal && !is_single &&
                     test.description.find("Double") != std::string::npos;
    bool is_mixed = !is_diagonal && !is_single &&
                    test.description.find("Mixed") != std::string::npos;

    if (almost_equal(hij_gpu, hij_cpu, 1e-12, 1e-14)) {
      std::cout << "  ✓ PASSED (match within tolerance)" << std::endl;
      passed++;
      if (is_diagonal)
        diagonal_passed++;
      else if (is_single)
        single_passed++;
      else if (is_double || is_mixed)
        double_passed++;
      else
        higher_passed++;
    } else {
      double abs_err = std::abs(hij_gpu - hij_cpu);
      double rel_err = abs_err / std::max(std::abs(hij_cpu), 1e-16);
      std::cout << "  ✗ FAILED" << std::endl;
      std::cout << "    Absolute error: " << abs_err << std::endl;
      std::cout << "    Relative error: " << rel_err << std::endl;
      failed++;
      if (is_diagonal)
        diagonal_failed++;
      else if (is_single)
        single_failed++;
      else if (is_double || is_mixed)
        double_failed++;
      else
        higher_failed++;
    }
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "Test Summary" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Overall: " << passed << " passed, " << failed << " failed"
            << std::endl;
  std::cout << "\nBy excitation type:" << std::endl;
  std::cout << "  Diagonal (zero):      " << diagonal_passed << " passed, "
            << diagonal_failed << " failed" << std::endl;
  std::cout << "  Single excitations:   " << single_passed << " passed, "
            << single_failed << " failed" << std::endl;
  std::cout << "  Double excitations:   " << double_passed << " passed, "
            << double_failed << " failed" << std::endl;
  std::cout << "  Higher-order:         " << higher_passed << " passed, "
            << higher_failed << " failed" << std::endl;
  std::cout << "========================================" << std::endl;

  if (failed > 0) {
    std::cout << "\n✗ CORRECTNESS TEST FAILED" << std::endl;
    std::cout << "GPU ComputeHij produces different results than CPU Hij"
              << std::endl;
    return 1;
  } else {
    std::cout << "\n✓ All tests passed" << std::endl;
    return 0;
  }
#else
  std::cout << "SKIPPED: Fe4S4 Minimal Test (GPU=0, test requires GPU support)"
            << std::endl;
  return 0;
#endif
}
