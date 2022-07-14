#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/command_line.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>
#include "../common/helper.h"
template<typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> &in) {
  std::vector<T> out;
  if (in.size() == 0)
    return out;
  const int cols = in[0].size();
  out.resize(in.size() * cols);
  for (uint64_t i = 0; i < in.size(); i++) {
    memcpy(&out[i * cols], in[i].data(), cols * sizeof(T));
  }
  return out;
}

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
    cutlass::gemm::GemmShape<128, 128, 4>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 4>;  // <- warp tile M = 64, N = 64, K = 32 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 8, N = 8, K = 4
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
                                                  8,     /*alignmentA*/
                                                  8,     /*alignmengB*/
                                                  cutlass::arch::OpMultiplyAdd,
                                                  cutlass::ComplexTransform::kNone,
                                                  cutlass::ComplexTransform::kNone,
                                                  true,  /*GatherA*/
                                                  true,  /*GatherB*/
                                                  true   /*ScatterD*/
                                                 >;

int main() {
  int m=4;
  int k=2;
  int n=4;

  std::vector<std::vector<float>> a = {{1, 1}, {0, 0}, {1, 1}, {0, 0}};
  std::vector<std::vector<float>> b = {{2}, {3}, {4}, {5}};
  std::vector<float> d = {0, 0, 0, 0};

  std::vector<int> a_idx = {0, 2};
  std::vector<int> b_idx = {1, 2};
  std::vector<int> d_idx = {0, 3};

  std::vector<float> a_flattened = flatten(a);
  std::vector<float> b_flattened = flatten(b);

  int index_size= a_idx.size();

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  cutlass::gemm::GemmCoord problem_size_real(problem_size.m(),
                                             index_size,
                                             index_size);
  float *d_a;
  float *d_b;
  float *d_c;
  float *d_d;

  cudaMalloc(&d_a, 8 * sizeof(float));
  cudaMalloc(&d_b, 4 * sizeof(float));
  cudaMalloc(&d_c, 4 * sizeof(float));
  cudaMalloc(&d_d, 4 * sizeof(float));

  cudaMemcpy(d_a, a_flattened.data(), 8 * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b_flattened.data(), 4 * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(d_c, 0, 4 * sizeof(float));
  cudaMemcpy(d_d, d.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);

  int *d_a_idx;
  int *d_b_idx;
  int *d_d_idx;

  cudaMalloc(&d_a_idx, 2 * sizeof(int));
  cudaMalloc(&d_b_idx, 2 * sizeof(int));
  cudaMalloc(&d_d_idx, 2 * sizeof(int));

  cudaMemcpy(d_a_idx, a_idx.data(), 2 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_idx, b_idx.data(), 2 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d_idx, d_idx.data(), 2 * sizeof(int), cudaMemcpyHostToDevice);
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(1);

  int split_k_slices = 1;
  cutlass::layout::RowMajor().capacity(problem_size.mk());
  Gemm::Arguments;
#if 0
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real, // <- problem size of matrix multiplication
      split_k_slices,    // <- k-dimension split factor
      {alpha, beta},     // <- alpha, beta
      d_a,               // <- reference to matrix A on device
      d_b,               // <- reference to matrix B on device
      d_c,               // <- reference to matrix C on device
      d_d,               // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(problem_size.mk()),
      cutlass::layout::RowMajor().capacity(
          cutlass::make_Coord(index_size, problem_size.n())),
      cutlass::layout::RowMajor().capacity(problem_size.mn()),
      cutlass::layout::RowMajor().capacity(problem_size.mn()),
      cutlass::layout::RowMajor().stride(),
      cutlass::layout::RowMajor().stride(),
      cutlass::layout::RowMajor().stride(),
      cutlass::layout::RowMajor().stride(),
      d_a_idx,  // <- pointer to index vector to gather A on device
      d_b_idx,  // <- pointer to index vector to gather B on device
      d_d_idx}; // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  //cutlass::Status status = gemm_op.can_implement(arguments);
  cutlass::Status status;
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  //status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  //status = gemm_op();
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);

  cudaMemcpy(d.data(), d_d, 4 * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 4; i++) {
    std::cout << d.at(i) << ',';
  }
#endif
}