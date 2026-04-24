// halving_doubling_allreduce.cu
// Rabenseifner halving-doubling all-reduce:
//   recursive-halving reduce-scatter + recursive-doubling all-gather.
// 2*log2(N) steps, bandwidth-optimal. Requires power-of-two n_ranks.

#include <assert.h>
#include <stdio.h>

#include "interface.h"



static int ilog2_exact(int x) {
    int r = 0;
    while ((1 << r) < x) r++;
    return r;
}

// halving-doubling all-reduce using NCCL send/recv
static void hd_allreduce_impl(
    const float* d_inbuf, float* d_outbuf, long input_size, ncclComm_t comm, cudaStream_t stream
) {
    int rank, n_ranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &n_ranks);
    assert((n_ranks & (n_ranks - 1)) == 0);  // power of two
    assert(input_size % n_ranks == 0);

    const int S = ilog2_exact(n_ranks);
    const long chunk_size = input_size / n_ranks;

    // copy input buffer to output buffer
    if (d_inbuf != d_outbuf)
        CUDA_CALL(cudaMemcpyAsync(
            d_outbuf, d_inbuf, input_size * sizeof(float), cudaMemcpyDeviceToDevice, stream
        ));

    // scratch for reduce-scatter: first step transfers input_size/2 elements
    float* temp_buf = nullptr;
    if (S > 0) CUDA_CALL(cudaMalloc(&temp_buf, (input_size / 2) * sizeof(float)));

    const int threads = 256;

    // --- REDUCE-SCATTER: recursive halving (MSB first) ---
    // At step d, split current block by bit h = S-1-d of the chunk index.
    // Partner has that bit flipped; we send the half we do NOT own, receive the
    // half we do own (from partner) and reduce it in-place.
    for (int d = 0; d < S; d++) {
        int h = S - 1 - d;
        int mask_bit = 1 << h;
        int partner = rank ^ mask_bit;
        long half_size = (long)mask_bit * chunk_size;      // 2^h chunks
        long kept_chunk = (long)(rank & ~(mask_bit - 1));  // my kept-half start
        long sent_chunk = kept_chunk ^ mask_bit;           // my sent-half start
        long kept_off = kept_chunk * chunk_size;
        long sent_off = sent_chunk * chunk_size;

        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend(d_outbuf + sent_off, half_size, ncclFloat, partner, comm, stream));
        NCCL_CALL(ncclRecv(temp_buf, half_size, ncclFloat, partner, comm, stream));
        NCCL_CALL(ncclGroupEnd());

        long blocks = (half_size + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, stream>>>(d_outbuf, temp_buf, half_size, kept_off);
        CUDA_CALL(cudaGetLastError());
    }

    // --- ALL-GATHER: recursive doubling (LSB first) ---
    // At step d, rank owns a contiguous block of 2^d chunks at (rank & ~(2^d-1)).
    // Partner (rank ^ 2^d) owns the sibling block. Exchange, then blocks merge.
    for (int d = 0; d < S; d++) {
        int mask_bit = 1 << d;
        int partner = rank ^ mask_bit;
        long block_size = (long)mask_bit * chunk_size;
        long my_off = (long)(rank & ~(mask_bit - 1)) * chunk_size;
        long partner_off = (long)(partner & ~(mask_bit - 1)) * chunk_size;

        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend(d_outbuf + my_off, block_size, ncclFloat, partner, comm, stream));
        NCCL_CALL(ncclRecv(d_outbuf + partner_off, block_size, ncclFloat, partner, comm, stream));
        NCCL_CALL(ncclGroupEnd());
    }

    CUDA_CALL(cudaStreamSynchronize(stream));
    if (S > 0) CUDA_CALL(cudaFree(temp_buf));
}



// interface function, runs for each MPI rank
void halving_doubling_allreduce(RunArgs* args) {
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


    // call halving-doubling all-reduce
    hd_allreduce_impl(d_inbuf, d_outbuf, input_size, comm, stream);


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
        hd_allreduce_impl(d_inbuf, d_outbuf, input_size, comm, stream);


    // benchmark
    double* deltas = (double*)malloc(args->n_iters * sizeof(double));
    for (int i = 0; i < args->n_iters; i++) {
        double t0 = get_time();
        hd_allreduce_impl(d_inbuf, d_outbuf, input_size, comm, stream);
        double t1 = get_time();
        deltas[i] = t1 - t0;
    }
    analyze_runtime(args, deltas);
    free(deltas);


    // cleanup
    CUDA_CALL(cudaFree(d_inbuf));
    CUDA_CALL(cudaFree(d_outbuf));
    CUDA_CALL(cudaStreamDestroy(stream));
}
