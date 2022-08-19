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

enum Type{kInt,kFloat,kDouble};
template<typename T>
__global__ void print(T*const p, size_t const len,Type t) {
  for(int i=0;i<len;i++) {
    if(kInt==t)
    printf("%d,",*(p+i));
    if(kFloat==t || kDouble==t)
    printf("%f,",*(p+i));
  }
  printf("\n");
}

int constexpr bsz = 4;
int constexpr x = 1;
int constexpr y = 1;
int constexpr z = 1;
int constexpr in_channel = 4;
int constexpr out_channel = 4;
int constexpr index_size = 2;

template <typename ElementAccumulator=float, typename ElementComputeEpilogue=float,
          typename ElementInputA=float, typename ElementInputB=float,
          typename ElementOutput=float, typename LayoutInputA=cutlass::layout::RowMajor, typename LayoutInputB=cutlass::layout::RowMajor,
          typename LayoutOutput=cutlass::layout::RowMajor, typename IdxT=int>
void gather_gemm_scatter(ElementInputA *const a, ElementInputB *const b,
                         ElementAccumulator *const c, ElementOutput *const d,
                         const int m, const int n, const int k,
                         const IdxT *a_indices, const IdxT *c_d_indices,
                         const int indices_size,
                         ElementComputeEpilogue const alpha,
                         ElementComputeEpilogue const beta) {
///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.

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
                                                  128 / cutlass::sizeof_bits<ElementInputA>::value,     /*alignmentA*/
                                                  128 / cutlass::sizeof_bits<ElementInputB>::value,     /*alignmengB*/
                                                  cutlass::arch::OpMultiplyAdd,
                                                  cutlass::ComplexTransform::kNone,
                                                  cutlass::ComplexTransform::kNone,
                                                  true,  /*GatherA*/
                                                  false,  /*GatherB*/
                                                  true   /*ScatterD*/
                                                 >;

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real({index_size, n, k});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, 
      problem_size_real,                  // <- problem size of matrix multiplication
      split_k_slices,                     // <- k-dimension split factor
      {alpha, beta},                      // <- alpha, beta
      a,             // <- reference to matrix A on device
      b,             // <- reference to matrix B on device
      c,             // <- reference to matrix C on device
      d,   // <- reference to matrix D on device
      cutlass::layout::RowMajor().capacity(cutlass::make_Coord(index_size, problem_size_real.k())),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(cutlass::make_Coord(index_size, problem_size_real.n())),
      cutlass::layout::RowMajor().capacity(cutlass::make_Coord(index_size, problem_size_real.n())),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,                             // <- pointer to index vector to gather A on device
      nullptr,       // <- pointer to index vector to gather B on device
      c_d_indices};      // <- pointer to index vector to scatter D on device

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

  status = gemm_op();
  cudaDeviceSynchronize();
  CUTLASS_CHECK(status);
}

int main() {
  srand(time(NULL));

    int constexpr m = bsz * x * y * z;
    int constexpr k = in_channel;
    int constexpr n = out_channel;

  // Initialize tensors using CUTLASS helper functions
    std::vector<float> tensor_a(m * k);
    std::vector<float> tensor_b(k * n);
    std::vector<float> tensor_c(m * n);
    std::vector<float> tensor_d_scattered(m * n, 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-7, 8);
    auto ran = [&dist, &gen]() { return trunc(dist(gen) * 1e5) / 1e5; };
    std::generate(tensor_a.begin(), tensor_a.end(), ran);
    std::generate(tensor_b.begin(), tensor_b.end(), ran);
    std::generate(tensor_c.begin(), tensor_c.end(), ran);
    std::vector<float> tensor_d_ref = tensor_d_scattered;

    std::vector<int> tensor_indices(index_size);
    std::vector<int> tensor_out_indices(index_size);

    // <- Fill tensor_b_indices on host with unique random integers
    std::vector<int> to_fill(m); // vector with ints.
    std::iota(std::begin(to_fill), std::end(to_fill),
              0); // Fill with 0, 1, ...., problem_size.n()
    std::random_shuffle(to_fill.begin(), to_fill.end());
    memcpy(tensor_indices.data(), to_fill.data(),
           index_size * sizeof(int));

    std::random_shuffle(to_fill.begin(), to_fill.end());
    memcpy(tensor_out_indices.data(), to_fill.data(),
           index_size * sizeof(int));


    std::stringstream fname;
    fname << "error_gather_GEMM_scatter_fusion.txt";
    std::cerr << "Dumping results in " << fname.str() << "\n";

    std::ofstream file(fname.str());

    file << "A =\n";
    for (auto &e : tensor_a)
      file << e << ',';

    file << "\nB =\n";
    for (auto &e : tensor_b)
      file << e << ',';

    file << "\nindices =\n";
    for (auto &e : tensor_indices)
      file << e << ',';

    file << "\nout_indices =\n";
    for (auto &e : tensor_out_indices)
      file << e << ',';

    file << "\nC =\n";
    for (auto &e : tensor_c)
      file << e << ',';

    file << "\nD =\n";
    for (auto &e : tensor_d_scattered)
      file << e << ',';

    file << "\n\nReference0 =\n";
    for (auto &e : tensor_d_ref)
      file << e << ',';
  // Copy data from host to GPU

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

  // Initialize alpha/beta for dot product computation

  int alpha=1;
  int beta=1;
  gather_gemm_scatter<float,float,float,float,float>(d_tensor_a,d_tensor_b,d_tensor_c,d_tensor_d_scattered,m,n,k,d_tensor_indices,d_tensor_out_indices,index_size,alpha,beta);
  //gather_gemm_scatter<float,float,float,float,float,cutlass::layout::RowMajor,cutlass::layout::RowMajor,cutlass::layout::RowMajor,int>(d_tensor_a,d_tensor_b,d_tensor_c,d_tensor_d_scattered,m,n,k,d_tensor_indices,d_tensor_out_indices,index_size,alpha,beta);

    for (int i = 0; i < index_size; ++i) {
      int a_row = tensor_indices.at(i);
      int c_d_row = tensor_out_indices.at(i);
      for (int j = 0; j < n; ++j) {

        for (int kk = 0; kk < k; ++kk) {
          tensor_d_ref.at(c_d_row * n + j) +=
              alpha * tensor_a.at(a_row * k + kk) *
              tensor_b.at(kk * n + j);
        }

        tensor_d_ref.at(c_d_row * n + j) +=
            (beta * tensor_c.at(c_d_row * n + j));
      }
    }

    cudaMemcpy(tensor_d_scattered.data(),d_tensor_d_scattered,tensor_d_scattered.size()*sizeof(float),cudaMemcpyDeviceToHost);

    bool passed = tensor_d_scattered == tensor_d_ref;

    file << "\n\nReference1 =\n";
    for (auto &e : tensor_d_ref)
      file << e << ',';

    file << "\nComputed =\n";
    for (auto &e : tensor_d_scattered)
      file << e << ',';

    file.close();
    if (!passed) {
      std::cout << "Failed!\n";

      return -1;
    } else {
      std::cout << "Passed!\n";
    }
}