#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

int constexpr bsz = 8;
int constexpr x = 100;
int constexpr y = 100;
int constexpr z = 100;
int constexpr index_size = 367;
int constexpr in_channel = 4;
int constexpr out_channel = 16;

template <typename ElementAccumulator=float, typename ElementComputeEpilogue=float,
          typename ElementInputA=float, typename ElementInputB=float,
          typename ElementOutput=float, typename LayoutInputA=cutlass::layout::RowMajor, typename LayoutInputB=cutlass::layout::RowMajor,
          typename LayoutOutput=cutlass::layout::RowMajor, typename IdxT=int>
void gather_gemm_scatter(ElementInputA *const A, ElementInputB *const B,
                         ElementAccumulator *const C, ElementOutput *const D,
                         const int m, const int n, const int k,
                         const IdxT *a_indices, const IdxT *c_d_indices,
                         const int indices_size,
                         ElementComputeEpilogue const alpha,
                         ElementComputeEpilogue const beta) {
  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
      cutlass::gemm::GemmShape<128, 128, 16>; // <- threadblock tile M = 128, N
                                              // = 128, K = 32
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp =
      cutlass::gemm::GemmShape<64, 64,
                               16>; // <- warp tile M = 64, N = 64, K = 32
  // This code section describes the size of MMA op
  using ShapeMMAOp =
      cutlass::gemm::GemmShape<16, 8, 8>; // <- MMA Op tile M = 8, N = 8, K = 4
  // 16, 8, 8 -> Turing
  // 16, 8, 16 -> Ampere

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

  // Define the epilogue operation as LinearCombination. This is approximately
  // equal to
  //
  //    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
  //
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, // <- data type of output matrix
      128 / cutlass::sizeof_bits<
                ElementOutput>::value, // <- this is the number of elements per
                                       // vectorized memory access. For half
                                       // precision, it's 8 elements. This
                                       // becomes the vector width of math
                                       // instructions in epilogue too
      ElementAccumulator,      // <- data type of accumulator
      ElementComputeEpilogue>; // <- data type for alpha in linear combination
                               // function

  // Number of pipelines you want to use
  constexpr int NumStages = 5;
  // Ampere -> 4/5
  // Turing -> 2

  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages,
      128 / cutlass::sizeof_bits<ElementInputA>::value, /*alignmentA*/
      128 / cutlass::sizeof_bits<ElementInputB>::value, /*alignmengB*/
      cutlass::arch::OpMultiplyAdd, cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone, true, /*GatherA*/
      false,                                  /*GatherB*/
      true                                    /*ScatterD*/
      >;
  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(m, n, k);

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real(indices_size, problem_size.n(),
                                             problem_size.k());

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,    // <- problem size of matrix multiplication
      split_k_slices,       // <- k-dimension split factor
      {alpha, beta},        // <- alpha, beta
      A,           // <- reference to matrix A on device
      B,           // <- reference to matrix B on device
      C,           // <- reference to matrix C on device
      D, // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(cutlass::make_Coord(index_size, problem_size_real.k())),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(cutlass::make_Coord(index_size, problem_size_real.n())),
      cutlass::layout::RowMajor().capacity(cutlass::make_Coord(index_size, problem_size_real.n())),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,      // <- pointer to index vector to gather A on device
      nullptr,               // <- pointer to index vector to gather B on device
      c_d_indices}; // <- pointer to index vector to scatter D on
                             // device

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
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

  status = gemm_op();
  CUTLASS_CHECK(status);
}

int main() {

    int constexpr m = bsz * x * y * z;
    int constexpr k = in_channel;
    int constexpr n = out_channel;
    std::vector<float> tensor_a(m * k);
    std::vector<float> tensor_b(k * n);
    std::vector<float> tensor_c(m * n);
    std::vector<float> tensor_d_scattered(m * n);

    std::generate(tensor_a.begin(), tensor_a.begin(),
                  []() { return rand() % 7; });
    std::generate(tensor_b.begin(), tensor_b.begin(),
                  []() { return rand() % 7; });
    std::generate(tensor_c.begin(), tensor_c.begin(),
                  []() { return rand() % 7; });
    std::generate(tensor_d_scattered.begin(), tensor_d_scattered.begin(),
                  []() { return rand() % 7; });

  std::vector<int> tensor_indices(index_size);
  std::vector<int> tensor_out_indices(index_size);


  // <- Fill tensor_b_indices on host with unique random integers
  std::vector<int> to_fill(m) ; // vector with ints.
  std::iota (std::begin(to_fill), std::end(to_fill), 0); // Fill with 0, 1, ...., problem_size.n()
  std::random_shuffle(to_fill.begin(), to_fill.end());
  memcpy(tensor_indices.data(), to_fill.data(), index_size * sizeof(int));

  std::random_shuffle(to_fill.begin(), to_fill.end());
  memcpy(tensor_out_indices.data(), to_fill.data(), index_size * sizeof(int));

  float *d_tensor_a;
  float *d_tensor_b;
  float *d_tensor_c;
  float *d_tensor_d_scattered;
  int *d_tensor_indices;
  int *d_tensor_out_indices;

  cudaMalloc(&d_tensor_a, tensor_a.size() * sizeof(float));
  cudaMalloc(&d_tensor_b, tensor_b.size() * sizeof(float));
  cudaMalloc(&d_tensor_c, tensor_c.size() * sizeof(float));
  cudaMalloc(&d_tensor_d_scattered,
             tensor_d_scattered.size() * sizeof(float));
  cudaMalloc(&d_tensor_indices, tensor_indices.size() * sizeof(int));
  cudaMalloc(&d_tensor_out_indices, tensor_out_indices.size() * sizeof(int));

  cudaDeviceSynchronize();

  cudaMemcpy(d_tensor_a, tensor_a.data(),
             tensor_a.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tensor_b, tensor_b.data(),
             tensor_b.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tensor_c, tensor_c.data(),
             tensor_c.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tensor_d_scattered, tensor_d_scattered.data(),
             tensor_d_scattered.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_tensor_indices, tensor_indices.data(),
             tensor_indices.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_tensor_out_indices, tensor_out_indices.data(),
             tensor_out_indices.size() * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  int alpha=1;
  int beta=1;
  gather_gemm_scatter(d_tensor_a,d_tensor_b,d_tensor_c,d_tensor_d_scattered,m,n,k,d_tensor_indices,d_tensor_out_indices,index_size,alpha,beta);
  cudaDeviceSynchronize();


  std::vector<float> tensor_d_ref(m * n, 0);

    for (int i = 0; i < index_size; ++i) {
      int a_row = tensor_indices.at(i);
      int c_d_row = tensor_out_indices.at(i);
      for (int j = 0; j < n; ++j) {

        for (int k = 0; k < k; ++k) {
          tensor_d_ref.at(c_d_row * n + j) +=
              alpha * tensor_a.at(a_row * k + k) *
              tensor_b.at(k * n + j);
        }

        tensor_d_ref.at(c_d_row * n + j) +=
            (beta * tensor_c.at(a_row * n + j));
      }
    }

    cudaMemcpy(tensor_d_scattered.data(),d_tensor_d_scattered,tensor_d_scattered.size()*sizeof(float),cudaMemcpyDeviceToHost);
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