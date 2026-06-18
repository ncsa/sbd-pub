#pragma once

#include <algorithm>  // For std::max
#include <cmath>      // For std::abs
#include <cstddef>    // For size_t
#include <functional> // For passing test functions to run_test
#include <iostream>
#include <stdexcept> // For throwing exceptions in TEST_ASSERT
#include <string>    // For error messages in TestResult
#include <vector>

/**
 * @brief A struct to hold the results of a single test case.
 */
struct TestResult {
  bool passed;
  std::string name;
  std::string error_msg;
};

/**
 * @brief An assertion macro that throws a runtime error if a condition is
 * false. This allows the run_test function to catch the failure.
 */
#define TEST_ASSERT(condition)                                                 \
  if (!(condition)) {                                                          \
    throw std::runtime_error("Assertion failed: " #condition);                 \
  }

/**
 * @brief Runs a given test function, catches any failures, and returns a
 * TestResult.
 * @param name The name of the test to be printed in the results.
 * @param test_func A function pointer or lambda to the test code.
 * @return A TestResult object indicating pass/fail status.
 */
inline TestResult run_test(const std::string &name,
                           std::function<void()> test_func) {
  try {
    test_func();
    return {true, name, ""};
  } catch (const std::exception &e) {
    return {false, name, e.what()};
  }
}

/**
 * @brief Checks if two double-precision numbers are approximately equal.
 * @param a First number.
 * @param b Second number.
 * @param tol Tolerance for comparison. Defaults to 1e-9.
 * @return True if the numbers are approximately equal, false otherwise.
 */
inline bool approx_equal(double a, double b, double tol = 1e-9) {
  double abs_a = std::abs(a);
  double abs_b = std::abs(b);
  double diff = std::abs(a - b);

  // If both are very small (near zero), use absolute tolerance
  if (abs_a < 1e-15 && abs_b < 1e-15) {
    return diff < tol;
  }

  // Otherwise use relative tolerance
  return diff / std::max(abs_a, abs_b) < tol;
}

/**
 * @brief Prints a determinant bitstring for debugging.
 * @param det The determinant vector.
 * @param bit_length The bit length of each word (e.g., 64).
 * @param L The number of alpha (or beta) orbitals.
 */
inline void print_bitstring(const std::vector<size_t> &det, size_t bit_length,
                            size_t L) {
  for (size_t i = 0; i < 2 * L; i++) {
    size_t block = i / bit_length;
    size_t bit_pos = i % bit_length;
    // Check block index to prevent out-of-bounds access
    if (block < det.size() && (det[block] & (size_t(1) << bit_pos))) {
      std::cout << "1";
    } else {
      std::cout << "0";
    }
  }
}
