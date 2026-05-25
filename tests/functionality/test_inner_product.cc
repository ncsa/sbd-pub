/**
 * @file test_inner_product.cc
 * @brief Unit and regression tests for InnerProduct.
 */

#include "sbd/sbd.h"
#include "utils.h"
#include <mpi.h>

using namespace sbd;

//=============================================================================
// InnerProduct Unit Tests
//=============================================================================

void test_InnerProduct_orthogonal() {
  MPI_Comm comm = MPI_COMM_WORLD;
  std::vector<double> X = {1.0, 0.0, 0.0, 0.0};
  std::vector<double> Y = {0.0, 1.0, 0.0, 0.0};
  double result;
  InnerProduct(X, Y, result, comm);
  TEST_ASSERT(approx_equal(result, 0.0));
}

void test_InnerProduct_parallel() {
  MPI_Comm comm = MPI_COMM_WORLD;
  std::vector<double> X = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> Y = {2.0, 4.0, 6.0, 8.0};
  double result;
  InnerProduct(X, Y, result, comm);
  TEST_ASSERT(approx_equal(result, 60.0));
}

void test_InnerProduct_normalized() {
  MPI_Comm comm = MPI_COMM_WORLD;
  double norm = std::sqrt(4.0);
  std::vector<double> X = {1.0 / norm, 1.0 / norm, 1.0 / norm, 1.0 / norm};
  double result;
  InnerProduct(X, X, result, comm);
  TEST_ASSERT(approx_equal(result, 1.0));
}

void test_InnerProduct_large() {
  MPI_Comm comm = MPI_COMM_WORLD;
  const size_t N = 100000;
  std::vector<double> X(N), Y(N);
  double expected = 0.0;
  for (size_t i = 0; i < N; i++) {
    X[i] = std::sin(i * 0.001);
    Y[i] = std::cos(i * 0.001);
    expected += X[i] * Y[i];
  }
  double result;
  InnerProduct(X, Y, result, comm);
  TEST_ASSERT(approx_equal(result, expected, 1e-10));
}

void test_InnerProduct_regression_Davidson() {
  MPI_Comm comm = MPI_COMM_WORLD;
  const size_t dim = 81796;
  std::vector<double> V1(dim);

  // Initialize the vector
  for (size_t i = 0; i < dim; i++) {
    V1[i] = std::sin(static_cast<double>(i) * 0.0001) *
            std::exp(-static_cast<double>(i) * 1e-6);
  }

  // Run the library functions
  double norm_check;
  Normalize(V1, norm_check, comm);

  double check_norm_sq;
  InnerProduct(V1, V1, check_norm_sq, comm);

  // Assert the final state
  const double tol = 1e-9;
  TEST_ASSERT(approx_equal(check_norm_sq, 1.0, tol));
  TEST_ASSERT(!std::isnan(check_norm_sq));
  TEST_ASSERT(!std::isinf(check_norm_sq));
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  std::vector<TestResult> results;

  results.push_back(
      run_test("InnerProduct - orthogonal", test_InnerProduct_orthogonal));
  results.push_back(
      run_test("InnerProduct - parallel", test_InnerProduct_parallel));
  results.push_back(
      run_test("InnerProduct - normalized", test_InnerProduct_normalized));
  results.push_back(run_test("InnerProduct - large", test_InnerProduct_large));
  results.push_back(run_test("InnerProduct - Davidson regression",
                             test_InnerProduct_regression_Davidson));

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
