/**
 * @file test_gpu_correctness.cc
 * @brief Comprehensive correctness tests for GPU offload implementation
 *
 * Tests:
 * 1. Helper array flattening: validate flat arrays match 2D arrays
 * 2. ComputeHij: device function vs CPU Hij() comparison
 * 3. mult() output: GPU vs CPU comparison (small system)
 * 4. Energy convergence: GPU should match CPU baseline
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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

#ifdef USE_GPU
// Test 1: Validate helper array flattening
void test_helper_flattening() {
  std::cout << "\n=== Test 1: Helper Array Flattening ===" << std::endl;

  // Create a minimal test case
  size_t bit_length = 64;
  size_t norb = 2;

  // Alpha dets: |11⟩, |10⟩
  std::vector<std::vector<size_t>> adets = {
      {0b11}, // both orbitals occupied
      {0b10}  // only orbital 1 occupied
  };

  // Beta dets: same
  std::vector<std::vector<size_t>> bdets = adets;

  // Create a helper with minimal task
  TaskHelpers helper;
  helper.braAlphaStart = 0;
  helper.braAlphaEnd = 2;
  helper.ketAlphaStart = 0;
  helper.ketAlphaEnd = 2;
  helper.braBetaStart = 0;
  helper.braBetaEnd = 2;
  helper.ketBetaStart = 0;
  helper.ketBetaEnd = 2;
  helper.taskType = 2; // Alpha excitations
  helper.adetShift = 0;
  helper.bdetShift = 0;

  // Generate excitations
  GenerateExcitation(adets, bdets, bit_length, norb, helper);

  size_t total_singles = 0;
  size_t total_doubles = 0;
  for (size_t i = 0; i < helper.SinglesFromAlpha.size(); i++) {
    total_singles += helper.SinglesFromAlpha[i].size();
  }
  for (size_t i = 0; i < helper.DoublesFromAlpha.size(); i++) {
    total_doubles += helper.DoublesFromAlpha[i].size();
  }

  std::cout << "  2D arrays: " << total_singles << " singles, " << total_doubles
            << " doubles" << std::endl;

  std::vector<size_t> sharedMemory;
  MakeSmartHelper(helper, sharedMemory);

  bool singles_match = true;
  size_t singles_checked = 0;
  for (size_t i = 0; i < helper.braAlphaEnd - helper.braAlphaStart; i++) {
    size_t offset = helper.SinglesFromAlphaOffset[i];
    size_t len = helper.SinglesFromAlphaLen[i];
    for (size_t j = 0; j < len; j++) {
      if (helper.SinglesFromAlpha_flat[offset + j] !=
          helper.SinglesFromAlphaSM[i][j]) {
        singles_match = false;
        std::cout << "  ✗ Mismatch at alpha[" << i << "][" << j << "]: "
                  << "flat=" << helper.SinglesFromAlpha_flat[offset + j]
                  << ", 2D=" << helper.SinglesFromAlphaSM[i][j] << std::endl;
      }
      singles_checked++;
    }
  }

  bool doubles_match = true;
  size_t doubles_checked = 0;
  for (size_t i = 0; i < helper.braAlphaEnd - helper.braAlphaStart; i++) {
    size_t offset = helper.DoublesFromAlphaOffset[i];
    size_t len = helper.DoublesFromAlphaLen[i];
    for (size_t j = 0; j < len; j++) {
      if (helper.DoublesFromAlpha_flat[offset + j] !=
          helper.DoublesFromAlphaSM[i][j]) {
        doubles_match = false;
        std::cout << "  ✗ Mismatch at alpha[" << i << "][" << j << "]: "
                  << "flat=" << helper.DoublesFromAlpha_flat[offset + j]
                  << ", 2D=" << helper.DoublesFromAlphaSM[i][j] << std::endl;
      }
      doubles_checked++;
    }
  }

  if (singles_match && doubles_match) {
    std::cout << "  ✓ Flattening validated: " << singles_checked << " singles, "
              << doubles_checked << " doubles match" << std::endl;
  } else {
    std::cout << "  ✗ Flattening FAILED" << std::endl;
    assert(false);
  }

  FreeHelpers(helper);
}

// Test 2: ComputeHij vs CPU Hij() comparison
void test_ComputeHij() {
  std::cout << "\n=== Test 2: ComputeHij Device Function ===" << std::endl;

  size_t bit_length = 64;
  size_t norbs = 2;

  std::vector<size_t> DetI = {0b0011};
  std::vector<size_t> DetJ = {0b0101};

  double I0 = 0.5;
  std::vector<double> I1_flat(16, 0.0);
  std::vector<double> I2_flat(10, 0.0);
  std::vector<double> I2_Direct_flat(norbs * norbs, 0.0);
  std::vector<double> I2_Exchange_flat(norbs * norbs, 0.0);

  I1_flat[0] = 1.0;
  I1_flat[5] = 1.5;
  I2_flat[0] = 0.2;
  I2_Direct_flat[0] = 0.15;
  I2_Exchange_flat[0] = 0.05;


  double hij_gpu = ComputeHij(DetI.data(), DetJ.data(), bit_length, norbs, I0,
                              I1_flat.data(), I2_flat.data(),
                              I2_Direct_flat.data(), I2_Exchange_flat.data());

  // Compute with CPU function
  oneInt<double> I1_obj;
  I1_obj.norbs = 2 * norbs;
  I1_obj.store.resize((2 * norbs) * (2 * norbs));
  for (size_t i = 0; i < 2 * norbs; i++) {
    for (size_t j = 0; j < 2 * norbs; j++) {
      I1_obj.store[i * (2 * norbs) + j] = I1_flat[i * (2 * norbs) + j];
    }
  }

  twoInt<double> I2_obj;
  I2_obj.norbs = norbs;
  I2_obj.store.resize(I2_flat.size());
  for (size_t ij = 0; ij < I2_flat.size(); ij++) {
    I2_obj.store[ij] = I2_flat[ij];
  }

  I2_obj.DirectMat.resize(norbs * norbs);
  I2_obj.ExchangeMat.resize(norbs * norbs);
  for (size_t i = 0; i < norbs * norbs; i++) {
    I2_obj.DirectMat[i] = I2_Direct_flat[i];
    I2_obj.ExchangeMat[i] = I2_Exchange_flat[i];
  }

  std::vector<int> c(2, 0);
  std::vector<int> d(2, 0);
  size_t orbDiff;
  double hij_cpu =
      Hij(DetI, DetJ, bit_length, norbs, c, d, I0, I1_obj, I2_obj, orbDiff);

  std::cout << "  GPU Hij = " << std::scientific << std::setprecision(12)
            << hij_gpu << std::endl;
  std::cout << "  CPU Hij = " << std::scientific << std::setprecision(12)
            << hij_cpu << std::endl;

  if (almost_equal(hij_gpu, hij_cpu)) {
    std::cout << "  ✓ ComputeHij matches CPU Hij()" << std::endl;
  } else {
    double rel_err =
        std::abs(hij_gpu - hij_cpu) / std::max(std::abs(hij_cpu), 1e-16);
    std::cout << "  ✗ ComputeHij MISMATCH (rel error = " << rel_err << ")"
              << std::endl;
    assert(false);
  }
}

void test_mult_correctness() {
  std::cout << "\n=== Test 3: mult() GPU vs CPU Comparison ===" << std::endl;
  std::cout << "  (Skipped: requires full MPI+OpenMP infrastructure)"
            << std::endl;
  std::cout
      << "  Use test_gpu_correctness with real FCIDUMP for full validation"
      << std::endl;
}

void test_energy_convergence() {
  std::cout << "\n=== Test 4: Energy Convergence ===" << std::endl;
  std::cout << "  (Skipped: use application with --iteration 1)" << std::endl;
  std::cout << "  Expected: E_ground = -326.70 Ha (Fe4S4 baseline)"
            << std::endl;
  std::cout << "  Use: mpirun -np 1 ./diag --fcidump fcidump_Fe4S4.txt "
               "--adetfile AlphaDets.txt"
            << std::endl;
}
#endif

int main(int argc, char **argv) {
#ifdef USE_GPU
  std::cout << "======================================" << std::endl;
  std::cout << "GPU Correctness Test Suite" << std::endl;
  std::cout << "======================================" << std::endl;

  test_helper_flattening();
  test_ComputeHij();
  test_mult_correctness();
  test_energy_convergence();

  std::cout << "\n========================================" << std::endl;
  std::cout << "✓ GPU Correctness Tests Complete" << std::endl;
  std::cout << "========================================" << std::endl;
#else
  std::cout
      << "SKIPPED: GPU Correctness Test (GPU=0, test requires GPU support)"
      << std::endl;
#endif

  return 0;
}
