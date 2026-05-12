// interface.h
// A common interface for all implementations of the ring algorithm

#pragma once

#include <cuda_runtime.h>
#include <nccl.h>



// define macros for running cuda and nccl commands with error checking
#define CUDA_CALL(cmd)                                                                       \
    do {                                                                                     \
        cudaError_t e = cmd;                                                                 \
        if (e != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA:%s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    } while (0)

#define NCCL_CALL(cmd)                                                                       \
    do {                                                                                     \
        ncclResult_t r = cmd;                                                                \
        if (r != ncclSuccess) {                                                              \
            fprintf(stderr, "NCCL:%s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    } while (0)


static constexpr int MAX_RANKS = 4;

/** Optional GPU-interval accumulation for stacked comm vs compute bars (-- env ALLREDUCE_BENCH_STACK).
 * Serialized (sync per bracket): valid for fractions of wall time, not for overlapped CUDA streams.
 */
typedef struct BenchStackAccum {
    cudaEvent_t comm0;
    cudaEvent_t comm1;
    cudaEvent_t comp0;
    cudaEvent_t comp1;
    int inited;
    double sum_comm_ms;
    double sum_compute_ms;
} BenchStackAccum;

typedef struct {
    // all-reduce arguments
    long input_size;
    ncclComm_t comm;
    // micro-batches for pipelined Ring / HD only (ignored by PAARD, naive)
    int n_batches;
    // benchmark & correctness arguments
    int n_warmup;
    int n_iters;
    float atol;
    bool* correct;
    double* avg_latency;
    double* std_latency;
    double* min_latency;
    double* max_latency;
    /** If non-NULL (stack benchmark mode): implementations fill with mean GPU ms per timed iteration across steps. */
    double* bench_avg_comm_us;
    double* bench_avg_compute_us;
} RunArgs;



// initialize buf[i] = 100*rank + 100*i/input_size
__global__ void init_input_kernel(float* buf, int rank, long input_size);

// element-wise add kernel: dest[i] += src[i]
__global__ void add_kernel(float* dest, const float* src, long n);

/** Read env knobs for add_kernel sleep, inter-node sim delay, etc. Call once per MPI rank after cudaSetDevice. */
void init_benchmark_knob_from_env(void);

void bench_accum_init(BenchStackAccum* b);
void bench_accum_destroy(BenchStackAccum* b);
void bench_stack_reset(BenchStackAccum* b);
void bench_stack_attach(BenchStackAccum* b);
void bench_stack_detach(void);
void bench_launch_add(long blocks, int threads, cudaStream_t stream, float* dest, const float* src, long n);

void ncclSendRecv(
    float* send_buf,
    float* recv_buf,
    size_t buf_sz,
    int rank,
    int send_rank,
    int recv_rank,
    ncclComm_t comm,
    cudaStream_t stream
);

// verify output[i] = 100*0 + 100*1 + ... 100*(n_ranks-1) + n_ranks*100*i/input_size
bool check_correctness(float* h_res, int rank, int n_ranks, long input_size, float atol);

// get current time in µs
double get_time();

// compute and record average, std, min, and max latency in µs
void analyze_runtime(RunArgs* args, double* deltas);



/** A common interface for the thread function that runs the ring algorithm for a rank.
 *
 * Every implementation must have the following signature and behavior:
 * - it should initialize the input using the init_input_kernel
 * - it should run the ring implementation and set correct using check_correctness
 * - it should run the ring implementation n_warmup times
 * - it should run the ring implementation n_iters more times and set avg_latency using get_time
 */
typedef void (*RingRunFunc)(RunArgs* args);



// TODO: add new implementations here
// void ring_nccl(RunArgs* args);
void ring_naive(RunArgs* args);
void ring_pipelined_nccl(RunArgs* args);
// void ring_pipelined_async(RunArgs* args);
void halving_doubling_allreduce(RunArgs* args);
void halving_doubling_pipelined(RunArgs* args);
void paard_nccl(RunArgs* args);
void paard_pipelined_nccl(RunArgs* args);
