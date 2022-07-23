/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// This example fuses gather before GEMM and scatter after GEMM into the same
// GEMM kernel.  Gather and scatter operation is controled by an index vector
// to select rows or columns from A, B, C or D matrices.
//
// Suppose, all matrices are column major.  The pseudo code of the fused kernel
// in this example is essentially
//
//    for (int i = 0; i < problem_size.m(); ++i) {
//      for (int j = 0; j < options.index_size; ++j) {
//        int b_c_d_col = tensor_indices.at({j, 0});
//
//        for (int k = 0; k < problem_size.k(); ++k) {
//            tensor_d_ref.at({i, b_c_d_col}) +=
//              alpha * tensor_a.at({i, k}) * tensor_b.at({k, b_c_d_col});
//        }
//      }
//    }
//
// Note that the index vector contains unique random integers with max to be N - 1
//
// The gather/scatter operation works best when we can still keep the biggest
// alignment. For example, when the matrix is row major, we select rows. When
// the matrix is column major, we selct columns.
//
// Not all the combination of gather and scatter are legal. For example, if A is
// row major and C/D is column major, we cannot gather A and scatter C/D at the
// same time.
//
// Also, we don't check the index value is legal and index array point is valid
// for the sake of the performance.
 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  cutlass::gemm::GemmCoord problem_size;
  int index_size;

  bool reference_check;
  int iterations;
  
  Options():
    help(false),
    problem_size({248, 1024, 1024}),
    index_size(240),
    reference_check(true),
    iterations(20) { }

  bool valid() {
    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());

    cmd.get_cmd_line_argument("index_size", index_size);
    
    cmd.get_cmd_line_argument("iterations", iterations);

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "36_gather_scatter_fusion example\n\n"
      << "  This example uses the CUTLASS Library to fuse gather/scatter into GEMM\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --index_size=<int>          size of N dimension index\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/36_gather_scatter_fusion/36_gather_scatter_fusion --m=1024 --n=512 --k=1024 \\\n"
      << "     --index_size=128\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = problem_size.product();
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = float;;             // <- data type of elements in input matrix A
using ElementInputB = float;;             // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices.
// Column Major for Matrix A, B and C.
//
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 16>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>;  // <- warp tile M = 64, N = 64, K = 32 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 8, N = 8, K = 4
// 16, 8, 8 -> Turing
// 16, 8, 16 -> Ampere

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// Define the epilogue operation as LinearCombination. This is approximately equal to
//
//    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
//
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,     // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
    ElementAccumulator,                                   // <- data type of accumulator
    ElementComputeEpilogue>;                              // <- data type for alpha in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 5;
// Ampere -> 4/5
// Turing -> 2

using Gemm = cutlass::gemm::device::GemmUniversal<ElementInputA,
                                                  LayoutInputA,
                                                  ElementInputB,
                                                  LayoutInputB,
                                                  ElementOutput,
                                                  LayoutOutput,
                                                  ElementAccumulator,
                                                  MMAOp,
                                                  SmArch,
                                                  ShapeMMAThreadBlock,
                                                  ShapeMMAWarp,
                                                  ShapeMMAOp,
                                                  EpilogueOp,
                                                  SwizzleThreadBlock,
                                                  NumStages,
                                                  4,     /*alignmentA*/
                                                  4,     /*alignmengB*/
                                                  cutlass::arch::OpMultiplyAdd,
                                                  cutlass::ComplexTransform::kNone,
                                                  cutlass::ComplexTransform::kNone,
                                                  true,  /*GatherA*/
                                                  false,  /*GatherB*/
                                                  true   /*ScatterD*/
                                                 >;

int run(Options &options) {

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size = options.problem_size;

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real(options.index_size,
                                             problem_size.n(),
                                             problem_size.k());

  // Initialize tensors using CUTLASS helper functions
#if 0
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N 
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_scattered(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
#endif

  std::vector<ElementInputA> tensor_a(problem_size.m()*problem_size.k());
  std::vector<ElementInputB> tensor_b(problem_size.k()*problem_size.n());
  std::vector<ElementOutput> tensor_c(problem_size.m()*problem_size.n());
  std::vector<ElementOutput> tensor_d_scattered(problem_size.m()*problem_size.n());

  std::generate(tensor_a.begin(),tensor_a.begin(),[](){return rand()%7;});
  std::generate(tensor_b.begin(),tensor_b.begin(),[](){return rand()%7;});
  std::generate(tensor_c.begin(),tensor_c.begin(),[](){return rand()%7;});
  std::generate(tensor_d_scattered.begin(),tensor_d_scattered.begin(),[](){return rand()%7;});

#if 0
  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a,
      1,
      ElementInputA(7),
      ElementInputA(-8),
      0);  // <- Fill matrix A on host with uniform-distribution random data

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b,
      1,
      ElementInputA(7),
      ElementInputA(-8),
      0);  // <- Fill matrix B on host with uniform-distribution random data

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c,
      1,
      ElementOutput(7),
      ElementOutput(-8),
      0);  // <- Fill matrix C on host with uniform-distribution random data

  cutlass::reference::host::TensorFill(
    tensor_d_scattered);  // <- fill matrix D on host with zeros

  cutlass::HostTensor<int, LayoutOutput> tensor_indices(
      {options.index_size, 1});  // <- Create scatter indices with dimensions val_len x 1
#endif

  std::vector<int> tensor_indices(options.index_size);

  // <- Fill tensor_b_indices on host with unique random integers
  std::vector<int> to_fill(problem_size.m()) ; // vector with ints.
  std::iota (std::begin(to_fill), std::end(to_fill), 0); // Fill with 0, 1, ...., problem_size.n()
  std::random_shuffle(to_fill.begin(), to_fill.end());
  memcpy(tensor_indices.data(), to_fill.data(), options.index_size * sizeof(int));

  // Copy data from host to GPU
#if 0
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_indices.sync_device();
  tensor_c.sync_device();
  tensor_d_scattered.sync_device();
#endif

  ElementInputA *d_tensor_a;
  ElementInputB *d_tensor_b;
  ElementOutput *d_tensor_c;
  ElementOutput *d_tensor_d_scattered;
  int *d_tensor_indices;

  cudaMalloc(&d_tensor_a, tensor_a.size() * sizeof(ElementInputA));
  cudaMalloc(&d_tensor_b, tensor_b.size() * sizeof(ElementInputB));
  cudaMalloc(&d_tensor_c, tensor_c.size() * sizeof(ElementOutput));
  cudaMalloc(&d_tensor_d_scattered,
             tensor_d_scattered.size() * sizeof(ElementOutput));
  cudaMalloc(&d_tensor_indices, tensor_indices.size() * sizeof(int));

  cudaDeviceSynchronize();

  cudaMemcpy(d_tensor_a, tensor_a.data(),
             tensor_a.size() * sizeof(ElementInputA), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tensor_b, tensor_b.data(),
             tensor_b.size() * sizeof(ElementInputB), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tensor_c, tensor_c.data(),
             tensor_c.size() * sizeof(ElementOutput), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tensor_d_scattered, tensor_d_scattered.data(),
             tensor_d_scattered.size() * sizeof(ElementOutput),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_tensor_indices, tensor_indices.data(),
             tensor_indices.size() * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  // Initialize alpha/beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(1);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, 
      problem_size_real,                  // <- problem size of matrix multiplication
      split_k_slices,                     // <- k-dimension split factor
      {alpha, beta},                      // <- alpha, beta
      d_tensor_a,             // <- reference to matrix A on device
      d_tensor_b,             // <- reference to matrix B on device
      d_tensor_c,             // <- reference to matrix C on device
      d_tensor_d_scattered,   // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(cutlass::make_Coord(options.index_size, problem_size.k())),
      cutlass::layout::RowMajor().capacity(problem_size.kn()),
      cutlass::layout::RowMajor().capacity(problem_size.mn()),
      cutlass::layout::RowMajor().capacity(problem_size.mn()),
      cutlass::layout::RowMajor().stride(),
      cutlass::layout::RowMajor().stride(),
      cutlass::layout::RowMajor().stride(),
      cutlass::layout::RowMajor().stride(),
      d_tensor_indices,                             // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      d_tensor_indices};      // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // CPU reference calculation
#if 0
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_ref(problem_size.mn());
  cutlass::reference::host::TensorFill(
    tensor_d_ref.host_view());  // <- Fill matrix D on host with zeros
#endif
  std::vector<ElementOutput> tensor_d_ref(problem_size.m() * problem_size.n(),
                                          0);

  status = gemm_op();
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);

  if (options.reference_check) {
    for (int i = 0; i < options.index_size; ++i) {
      int a_c_d_row = tensor_indices.at(i);
      for (int j = 0; j < problem_size.n(); ++j) {

        for (int k = 0; k < problem_size.k(); ++k) {
          tensor_d_ref.at(a_c_d_row * problem_size.n() + j) +=
              alpha * tensor_a.at(a_c_d_row * problem_size.k() + k) *
              tensor_b.at(k * problem_size.n() + j);
        }

        tensor_d_ref.at(a_c_d_row * problem_size.n() + j) +=
            (beta * tensor_c.at(a_c_d_row * problem_size.n() + j));
      }
    }

    // Copy output data from CUTLASS and reference kernel to host for comparison
    #if 0
    tensor_d_scattered.sync_host();
    #endif
    cudaMemcpy(tensor_d_scattered.data(),d_tensor_d_scattered,tensor_d_scattered.size()*sizeof(ElementOutput),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  
    bool passed = tensor_d_scattered == tensor_d_ref;

    if (!passed) {
      std::cout << "Failed!\n";

      std::stringstream fname;
      fname << "error_gather_GEMM_scatter_fusion.txt";
      std::cerr << "Dumping results in " << fname.str() << "\n";

      std::ofstream file(fname.str());

      file 
        << "A =\n" << tensor_a.data()
        << "\nB =\n" << tensor_b.data()
        << "\nindices =\n" << tensor_indices.data()
        << "\nC =\n" << tensor_c.data()
        << "\n\nReference =\n" << tensor_d_ref.data()
        << "\nComputed =\n" << tensor_d_scattered.data();
      return -1;
    } else {
      std::cout << "Passed!\n";
    }
  }

  // Result structure
  Result result;

  //
  // Construct events
  //

  cudaEvent_t events[2];

  for (auto & event : events) {
    result.error = cudaEventCreate(&event);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
  }

  // Record an event at the start of a series of GEMMs
  result.error = cudaEventRecord(events[0]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  //
  // Run profiling loop
  //

  for (int iter = 0; iter < options.iterations; ++iter) {
    // Launch initialized CUTLASS kernel
    status = gemm_op();
    CUTLASS_CHECK(status);
  }

  //
  // Stop profiling loop
  //

  // Record an event when the GEMMs are complete
  result.error = cudaEventRecord(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Wait for work on the device to complete.
  result.error = cudaEventSynchronize(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Measure elapsed runtime
  float runtime_ms = 0;
  result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Compute average runtime and GFLOPs.
  result.runtime_ms = double(runtime_ms) / double(options.iterations);
  result.gflops = options.gflops(result.runtime_ms / 1000.0);

  // Cleanup
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }

  std::cout << "Runtime: " << result.runtime_ms << " ms\n";
  std::cout << " GFLOPs: " << result.gflops << "\n";

  return 0;
}

int main(int argc, const char ** argv) {
  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync are first available in CUDA 11.0.
  //
  // CUTLASS must be compiled with CUDA 11 Toolkit to run Conv2dFprop examples.
  if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  if (!(props.major > 8 || (props.major == 8 && props.minor >= 0))) {
    std::cerr << "Ampere Tensor Ops must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    return 0;
  }

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << "\n";
    return 0;
  }

  if (!options.valid()) {
    std::cerr << "Invalid problem." << "\n";
    return -1;
  }

  return run(options);
}
