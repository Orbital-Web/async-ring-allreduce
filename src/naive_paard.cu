// naive_paard.cu
// Implements naive paard all-reduce with ncclSend/ncclRecv.

#include <assert.h>
#include <stdio.h>

#include <utility>

#include "interface.h"

// PAARD only works for dragonfly topology with DF(N,2N,N), i.e.,
// N nodes per router, 2N routers per group, N groups total
// given our limited compute, we can only run PAARD for DF(1,2,1) with 6 nodes so we hardcode it
constexpr int N = 1;
constexpr int P = N;
constexpr int A = 2 * N;
constexpr int H = N;

constexpr int group_size = P * A;
constexpr int n_groups = 1 + A * H;
constexpr int n_ranks = group_size * n_groups;



// static void print_debug(
//     int rank, float* d_outbuf, long input_size, long chunk_sz, ncclComm_t comm, cudaStream_t
//     stream
// ) {
//     float* temp_buf = nullptr;
//     size_t temp_sz = input_size * n_ranks * sizeof(float);
//     CUDA_CALL(cudaMalloc(&temp_buf, temp_sz));

//     NCCL_CALL(ncclAllGather(d_outbuf, temp_buf, input_size, ncclFloat, comm, stream));
//     CUDA_CALL(cudaStreamSynchronize(stream));

//     if (rank == 0) {
//         float* h_temp_buf = (float*)malloc(temp_sz);
//         CUDA_CALL(cudaMemcpy(h_temp_buf, temp_buf, temp_sz, cudaMemcpyDeviceToHost));

//         // display each rank's buffer vertically
//         for (long i = 0; i < input_size; i++) {
//             for (int r = 0; r < n_ranks; r++) {
//                 printf("%7.2f ", h_temp_buf[r * input_size + i]);
//                 if ((r + 1) % group_size == 0) printf(" ");
//             }
//             printf("\n");
//             if ((i + 1) % chunk_sz == 0) printf("\n");
//         }
//         printf("\n\n");
//         fflush(stdout);

//         free(h_temp_buf);
//     }

//     CUDA_CALL(cudaFree(temp_buf));
// }



// paard all-reduce
static void paard_allreduce(
    const float* d_inbuf, float* d_outbuf, long input_size, ncclComm_t comm, cudaStream_t stream
) {
    // get rank and group info
    int rank;
    ncclCommUserRank(comm, &rank);
    int group = rank / group_size;
    int local_rank = rank % group_size;

    // copy input buffer to output buffer
    if (d_inbuf != d_outbuf)
        CUDA_CALL(cudaMemcpyAsync(
            d_outbuf, d_inbuf, input_size * sizeof(float), cudaMemcpyDeviceToDevice, stream
        ));

    // determine chunk sizes and allocate temporary buffer
    assert(input_size % n_ranks == 0);
    long chunk_sz = input_size / n_groups;
    long sbchunk_sz = input_size / n_ranks;

    float* temp_buf = nullptr;
    CUDA_CALL(cudaMalloc(&temp_buf, chunk_sz * sizeof(float)));


    // --- STEP 1: INTERNAL REDUCE-SCATTER ---
    int lr_send = group * group_size + (local_rank + 1) % group_size;
    int lr_recv = group * group_size + (local_rank - 1 + group_size) % group_size;

    {
        int lch_send = (rank + group) % group_size;
        int lch_recv = (rank + group + 1) % group_size;
        if (lch_send >= group) lch_send++;
        if (lch_recv >= group) lch_recv++;

        long off_send = lch_send * chunk_sz;
        long off_recv = lch_recv * chunk_sz;

        ncclSendRecv(d_outbuf + off_send, temp_buf, chunk_sz, rank, lr_send, lr_recv, comm, stream);

        const int threads = 256;
        long blocks = (chunk_sz + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, stream>>>(d_outbuf + off_recv, temp_buf, chunk_sz);
        CUDA_CALL(cudaGetLastError());
    }


    // --- STEP 2: GLOBAL REDUCE ---
    int gr_send = ((n_ranks + group) * (n_groups - 1) - 1 + rank * n_groups) % n_ranks;
    int gr_recv = gr_send;

    {
        int gch_send = (n_ranks - 1 - rank) % n_groups;
        int gch_recv = group;

        long off_send = gch_send * chunk_sz;
        long off_recv = gch_recv * chunk_sz;

        ncclSendRecv(d_outbuf + off_send, temp_buf, chunk_sz, rank, gr_send, gr_recv, comm, stream);

        const int threads = 256;
        long blocks = (chunk_sz + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, stream>>>(d_outbuf + off_recv, temp_buf, chunk_sz);
        CUDA_CALL(cudaGetLastError());
    }


    // --- STEP 3: INTERNAL REDUCE-SCATTER ---
    {
        int lsbch_send = (rank + 1) % group_size + group * group_size;
        int lsbch_recv = rank;

        long off_send = lsbch_send * sbchunk_sz;
        long off_recv = lsbch_recv * sbchunk_sz;

        ncclSendRecv(
            d_outbuf + off_send, temp_buf, sbchunk_sz, rank, lr_send, lr_recv, comm, stream
        );

        // reduce
        const int threads = 256;
        long blocks = (sbchunk_sz + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, stream>>>(d_outbuf + off_recv, temp_buf, sbchunk_sz);
        CUDA_CALL(cudaGetLastError());
    }

    // --- STEP 4: INTERNAL ALL-GATHER ---
    {
        int lsbch_send = rank;
        int lsbch_recv = (rank + 1) % group_size + group * group_size;

        long off_send = lsbch_send * sbchunk_sz;
        long off_recv = lsbch_recv * sbchunk_sz;

        ncclSendRecv(
            d_outbuf + off_send,
            d_outbuf + off_recv,
            sbchunk_sz,
            rank,
            lr_send,
            lr_recv,
            comm,
            stream
        );
    }

    // --- STEP 5: GLOBAL GATHER ---
    {
        int gch_send = group;
        int gch_recv = (n_ranks - 1 - rank) % n_groups;

        long off_send = gch_send * chunk_sz;
        long off_recv = gch_recv * chunk_sz;

        ncclSendRecv(
            d_outbuf + off_send, d_outbuf + off_recv, chunk_sz, rank, gr_send, gr_recv, comm, stream
        );
    }

    // --- STEP 6: INTERNAL ALL-GATHER ---
    {
        int lch_send = (rank + group + 1) % group_size;
        int lch_recv = (rank + group) % group_size;
        if (lch_send >= group) lch_send++;
        if (lch_recv >= group) lch_recv++;

        long off_send = lch_send * chunk_sz;
        long off_recv = lch_recv * chunk_sz;

        ncclSendRecv(
            d_outbuf + off_send, d_outbuf + off_recv, chunk_sz, rank, lr_send, lr_recv, comm, stream
        );
    }

    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaFree(temp_buf));
}



// interface function, runs for each rank
void paard_nccl(RunArgs* args) {
    long input_size = args->input_size;
    ncclComm_t comm = args->comm;
    int rank, _n_ranks, device;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &_n_ranks);
    ncclCommCuDevice(comm, &device);
    assert(_n_ranks == n_ranks);


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


    // call paard all-reduce
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
