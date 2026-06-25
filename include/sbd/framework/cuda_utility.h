/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SBD_FRAMEWORK_CUDA_UTILITY_H
#define SBD_FRAMEWORK_CUDA_UTILITY_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#include <cuda.h>
#endif

// -----------------------------------------------------------------------------
// CUDA error check
// -----------------------------------------------------------------------------
#ifndef SBD_CHECK_CUDA
#define SBD_CHECK_CUDA(cmd)                                              \
    do {                                                                 \
        cudaError_t e = (cmd);                                           \
        if (e != cudaSuccess) {                                          \
            std::fprintf(stderr,                                         \
                "CUDA error %s at %s:%d\n",                              \
                cudaGetErrorString(e), __FILE__, __LINE__);              \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)
#endif

#endif // SBD_FRAMEWORK_CUDA_UTILITY_H
