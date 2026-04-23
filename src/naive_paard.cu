// naive_paard.cu
// Implements naive paard all-reduce with ncclSend/ncclRecv.

#include <assert.h>
#include <stdio.h>

#include <utility>

#include "interface.h"



static std::pair<long, long> get_offset1(
    int step, int rank, int n_ranks, int n_groups, long chunk_size
) {
    // by the end, rank r should have reduced chunk (n_ranks - 1 - r) % n_groups
    assert(n_groups <= n_ranks);
    assert(step >= 0);
    assert(step < n_groups - 1);
    long send_chunk = (2 * n_ranks - 2 - rank - step) % n_groups;
    long recv_chunk = (2 * n_ranks - 3 - rank - step) % n_groups;
    assert(send_chunk >= 0);
    assert(recv_chunk >= 0);
    return {send_chunk * chunk_size, recv_chunk * chunk_size};
}

static std::pair<long, long> get_offset2(
    int rank, int group, int n_ranks, int n_groups, long chunk_size
) {
    long send_chunk = (n_ranks - 1 - rank) % n_groups;
    long recv_chunk = group;
    return {send_chunk * chunk_size, recv_chunk * chunk_size};
}

static std::pair<long, long> get_offset3(
    int step,
    int local_rank,
    int group,
    int group_size,
    int n_groups,
    long chunk_size,
    long sbchunk_size
) {
    // by the end, rank r should have the reduced sbchunk r
    assert(step >= 0);
    assert(step < n_groups - 1);
    long send_chunk = (2 * group_size - 1 + local_rank - step) % group_size;
    long recv_chunk = (2 * group_size - 2 + local_rank - step) % group_size;
    assert(send_chunk >= 0);
    assert(recv_chunk >= 0);
    long base_offset = group * chunk_size;
    return {send_chunk * sbchunk_size + base_offset, recv_chunk * sbchunk_size + base_offset};
}



// element-wise add kernel: dest[i + offset] += src[i]
static __global__ void add_kernel(float* dest, const float* src, long offset, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dest[offset + idx] += src[idx];
}



// paard all-reduce
static void paard_allreduce(
    const float* d_inbuf, float* d_outbuf, long input_size, ncclComm_t comm, cudaStream_t stream
) {
    // get rank and number of ranks
    int rank, n_ranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &n_ranks);

    // get group info
    static constexpr int group_size = 2;  // no. ranks per group
    assert(n_ranks % group_size == 0);
    int n_groups = n_ranks / group_size;
    int group = rank / group_size;
    int local_rank = rank % group_size;

    // copy input buffer to output buffer
    if (d_inbuf != d_outbuf)
        CUDA_CALL(cudaMemcpyAsync(
            d_outbuf, d_inbuf, input_size * sizeof(float), cudaMemcpyDeviceToDevice, stream
        ));

    // compute chunk size and allocate temporary receive buffer
    assert(input_size % n_groups == 0);
    assert(input_size % n_ranks == 0);
    long chunk_size = input_size / n_groups;
    long sbchunk_size = input_size / n_ranks;
    float* temp_buf = nullptr;
    CUDA_CALL(cudaMalloc(&temp_buf, chunk_size * sizeof(float)));

    // --- STEP 1: INTERNAL REDUCE-SCATTER ---
    int local_next = group * group_size + (local_rank + 1) % group_size;
    int local_prev = group * group_size + (local_rank - 1 + group_size) % group_size;
    for (int step = 0; step < group_size - 1; step++) {
        auto [send_off, recv_off] = get_offset1(step, rank, n_ranks, n_groups, chunk_size);
        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend(d_outbuf + send_off, chunk_size, ncclFloat, local_next, comm, stream));
        NCCL_CALL(ncclRecv(temp_buf, chunk_size, ncclFloat, local_prev, comm, stream));
        NCCL_CALL(ncclGroupEnd());

        // reduce
        const int threads = 256;
        long blocks = (chunk_size + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, stream>>>(d_outbuf, temp_buf, recv_off, chunk_size);
        CUDA_CALL(cudaGetLastError());
    }

    // --- STEP 2: GLOBAL REDUCE ---
    int global_nbr = ((n_ranks + group) * (n_groups - 1) - 1 * rank * n_groups) % n_ranks;
    {
        auto [send_off, recv_off] = get_offset2(rank, group, n_ranks, n_groups, chunk_size);
        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend(d_outbuf + send_off, chunk_size, ncclFloat, global_nbr, comm, stream));
        NCCL_CALL(ncclRecv(temp_buf, chunk_size, ncclFloat, global_nbr, comm, stream));
        NCCL_CALL(ncclGroupEnd());

        // reduce
        const int threads = 256;
        long blocks = (chunk_size + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, stream>>>(d_outbuf, temp_buf, recv_off, chunk_size);
        CUDA_CALL(cudaGetLastError());
    }

    // --- STEP 3: INTERNAL REDUCE-SCATTER ---
    for (int step = 0; step < group_size - 1; step++) {
        auto [send_off, recv_off]
            = get_offset3(step, local_rank, group, group_size, n_groups, chunk_size, sbchunk_size);
        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend(d_outbuf + send_off, sbchunk_size, ncclFloat, local_next, comm, stream));
        NCCL_CALL(ncclRecv(temp_buf, sbchunk_size, ncclFloat, local_prev, comm, stream));
        NCCL_CALL(ncclGroupEnd());

        // reduce
        const int threads = 256;
        long blocks = (sbchunk_size + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, stream>>>(d_outbuf, temp_buf, recv_off, sbchunk_size);
        CUDA_CALL(cudaGetLastError());
    }


    // --- TODO: ALL-GATHER ---

    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaFree(temp_buf));
}



// interface function, runs for each rank
void paard_nccl(RunArgs* args) {
    long input_size = args->input_size;
    ncclComm_t comm = args->comm;
    int rank, n_ranks, device;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &n_ranks);
    ncclCommCuDevice(comm, &device);


    // initialize CUDA stream
    CUDA_CALL(cudaSetDevice(device));
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));


    // initialize input and output
    float* d_inbuf = nullptr;
    CUDA_CALL(cudaMalloc(&d_inbuf, input_size * sizeof(float)));

    const int threads = 256;
    long blocks = (input_size + threads - 1) / threads;
    init_input_kernel<<<blocks, threads, 0, stream>>>(d_inbuf, rank, input_size);
    CUDA_CALL(cudaGetLastError());

    float* d_outbuf = nullptr;
    CUDA_CALL(cudaMalloc(&d_outbuf, input_size * sizeof(float)));


    // call ring all-reduce
    paard_allreduce(d_inbuf, d_outbuf, input_size, comm, stream);


    // copy back result to host and verify output, short circuit if incorrect
    float* h_res = (float*)malloc(input_size * sizeof(float));
    CUDA_CALL(cudaMemcpy(h_res, d_outbuf, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    *(args->correct) = check_correctness(h_res, rank, n_ranks, input_size, args->atol);
    free(h_res);

    if (!*(args->correct)) {
        CUDA_CALL(cudaFree(d_inbuf));
        CUDA_CALL(cudaFree(d_outbuf));
        CUDA_CALL(cudaStreamDestroy(stream));
        return;
    }


    // warmup
    for (int i = 0; i < args->n_warmup; i++)
        paard_allreduce(d_inbuf, d_outbuf, input_size, comm, stream);


    // benchmark
    double* deltas = (double*)malloc(args->n_iters * sizeof(double));
    for (int i = 0; i < args->n_iters; i++) {
        double t0 = get_time();
        paard_allreduce(d_inbuf, d_outbuf, input_size, comm, stream);
        double t1 = get_time();
        deltas[i] = t1 - t0;
    }
    analyze_runtime(args, deltas);
    free(deltas);


    // cleanup
    CUDA_CALL(cudaFree(d_inbuf));
    CUDA_CALL(cudaFree(d_outbuf));
    CUDA_CALL(cudaStreamDestroy(stream));
    return;
}
