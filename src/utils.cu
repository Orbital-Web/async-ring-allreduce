// utils.cu

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "interface.h"

__global__ void init_input_kernel(float* buf, int rank, long input_size) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) buf[idx] = 100.0f * rank + idx * 100.0f / input_size;
}

bool check_correctness(float* h_res, int rank, int n_ranks, long input_size, float atol) {
    int sum_ranks = n_ranks * (n_ranks - 1) * 50;
    for (long i = 0; i < input_size; i++) {
        float expected = (float)sum_ranks + (float)n_ranks * 100.0f * i / input_size;
        float got = h_res[i];
        float diff = fabsf(got - expected);
        if (diff > atol) {
            fprintf(
                stderr,
                "Rank %d: verification FAILED, mismatch at idx %d: got %f expected %f (diff %f)\n",
                rank,
                i,
                got,
                expected,
                diff
            );
            return false;
        }
    }
    return true;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

// in-stream delay: one thread spins on __nanosleep so the penalty is enqueued
// on the CUDA stream alongside NCCL ops. preserves async overlap between
// compute and comm streams (unlike a host-side usleep which would require a
// cudaStreamSynchronize and serialize everything).
// __nanosleep caps at ~1 ms per call on sm_70+, so we chunk.
__global__ void delay_kernel(long total_ns) {
    const unsigned int chunk_ns = 100000;  // 100 us per __nanosleep call
    long remaining = total_ns;
    while (remaining > 0) {
        unsigned int s = remaining > chunk_ns ? chunk_ns : (unsigned int)remaining;
        __nanosleep(s);
        remaining -= s;
    }
}

void maybe_penalize_internode(cudaStream_t stream, long bytes) {
    // read penalty once, cache in statics (all ranks read the same env vars)
    static long penalty_us = -1;           // GLOBAL_PENALTY_US:  fixed-latency component (us)
    static double inv_bw_ns_per_byte = -1; // derived from GLOBAL_BW_GBPS: 1 B / 1 GB/s = 1 ns/B
    if (penalty_us < 0) {
        const char* env = getenv("GLOBAL_PENALTY_US");
        penalty_us = (env && env[0] != '\0') ? atol(env) : 0;

        const char* bw_env = getenv("GLOBAL_BW_GBPS");
        double bw_gbps = (bw_env && bw_env[0] != '\0') ? atof(bw_env) : 0.0;
        inv_bw_ns_per_byte = (bw_gbps > 0.0) ? (1.0 / bw_gbps) : 0.0;
    }

    long total_ns = penalty_us * 1000L
                  + (long)(inv_bw_ns_per_byte * (double)bytes);
    if (total_ns <= 0) return;

    delay_kernel<<<1, 1, 0, stream>>>(total_ns);
}

void analyze_runtime(RunArgs* args, double* deltas) {
    const int n_iters = args->n_iters;

    double sum_latency = 0.0;
    double min_latency = deltas[0];
    double max_latency = deltas[0];
    for (int i = 0; i < n_iters; i++) {
        double t = deltas[i];
        sum_latency += t;
        if (t < min_latency) min_latency = t;
        if (t > max_latency) max_latency = t;
    }

    double avg_latency = sum_latency / n_iters;
    double sum_std = 0.0;
    for (int i = 0; i < n_iters; i++)
        sum_std += (deltas[i] - avg_latency) * (deltas[i] - avg_latency);

    *(args->avg_latency) = avg_latency;
    *(args->std_latency) = sqrt(sum_std / n_iters);
    *(args->min_latency) = min_latency;
    *(args->max_latency) = max_latency;
}
