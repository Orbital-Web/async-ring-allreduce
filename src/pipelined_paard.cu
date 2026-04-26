// pipelined_paard.cu
// Implements pipelined paard all-reduce with ncclSend/ncclRecv.

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



// paard all-reduce
static void paard_allreduce(
    const float* d_inbuf,
    float* d_outbuf,
    long input_size,
    ncclComm_t comm,
    cudaStream_t streams[group_size]
) {
    // get rank and group info
    int rank;
    ncclCommUserRank(comm, &rank);
    int group = rank / group_size;
    int local_rank = rank % group_size;

    // copy input buffer to output buffer
    if (d_inbuf != d_outbuf)
        CUDA_CALL(cudaMemcpyAsync(
            d_outbuf, d_inbuf, input_size * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]
        ));

    // determine chunk sizes and allocate temporary buffer
    assert(input_size % n_ranks == 0);
    long chunk_sz = input_size / n_groups;
    long sbchunk_sz = input_size / n_ranks;

    // we pipeline by a factor of group_size so each step sends the same (input_size / n_ranks) msg
    float* temp_bufs[group_size];
    for (int i = 0; i < group_size; i++)
        CUDA_CALL(cudaMalloc(&temp_bufs[i], sbchunk_sz * sizeof(float)));


    // --- STEP 1: INTERNAL REDUCE-SCATTER ---
    int lr_send = group * group_size + (local_rank + 1) % group_size;
    int lr_recv = group * group_size + (local_rank - 1 + group_size) % group_size;

    {
        // NOTE: normally we'd have the head, loop(1,group_size-1), and tail
        // but since group_size = 2, it's a little silly to write an empty for loop
        // just imagine for larger dragonfly, instead of using [0] or [1], we'd use the loop index
        int lch_send = (rank + group) % group_size;
        int lch_recv = (rank + group + 1) % group_size;
        if (lch_send >= group) lch_send++;
        if (lch_recv >= group) lch_recv++;

        long off_send = lch_send * chunk_sz;
        long off_recv = lch_recv * chunk_sz;

        ncclSendRecv(
            d_outbuf + off_send, temp_bufs[0], sbchunk_sz, rank, lr_send, lr_recv, comm, streams[0]
        );

        // reduce and send next microbatch simultaneously
        const int threads = 256;
        long blocks = (sbchunk_sz + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, streams[0]>>>(
            d_outbuf + off_recv, temp_bufs[0], sbchunk_sz
        );
        CUDA_CALL(cudaGetLastError());

        off_send += sbchunk_sz;
        off_recv += sbchunk_sz;

        ncclSendRecv(
            d_outbuf + off_send, temp_bufs[1], sbchunk_sz, rank, lr_send, lr_recv, comm, streams[1]
        );

        // this gets pipelined with the global communication over stream[0] in step 2
        add_kernel<<<blocks, threads, 0, streams[1]>>>(
            d_outbuf + off_recv, temp_bufs[1], sbchunk_sz
        );
        CUDA_CALL(cudaGetLastError());
    }


    // --- STEP 2: GLOBAL REDUCE ---
    int gr_send = ((n_ranks + group) * (n_groups - 1) - 1 + rank * n_groups) % n_ranks;
    int gr_recv = gr_send;

    {
        // NOTE: again, this would normally be written as a loop
        int gch_send = (n_ranks - 1 - rank) % n_groups;
        int gch_recv = group;

        long off_send = gch_send * chunk_sz;
        long off_recv = gch_recv * chunk_sz;

        ncclSendRecv(
            d_outbuf + off_send, temp_bufs[0], sbchunk_sz, rank, gr_send, gr_recv, comm, streams[0]
        );

        // reduce and send next microbatch simultaneously
        const int threads = 256;
        long blocks = (sbchunk_sz + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, streams[0]>>>(
            d_outbuf + off_recv, temp_bufs[0], sbchunk_sz
        );
        CUDA_CALL(cudaGetLastError());

        off_send += sbchunk_sz;
        off_recv += sbchunk_sz;

        ncclSendRecv(
            d_outbuf + off_send, temp_bufs[1], sbchunk_sz, rank, gr_send, gr_recv, comm, streams[1]
        );

        // this gets pipelined with the local communication over stream[0] in step 3
        add_kernel<<<blocks, threads, 0, streams[1]>>>(
            d_outbuf + off_recv, temp_bufs[1], sbchunk_sz
        );
        CUDA_CALL(cudaGetLastError());
    }


    // --- STEP 3: INTERNAL REDUCE-SCATTER ---
    {
        // NOTE: same thing with the for loop
        // normally, the (local_ranks + 1) == loop index will take turn sending
        int lsbch_send = (rank + 1) % group_size + group * group_size;
        int lsbch_recv = rank;

        long off_send = lsbch_send * sbchunk_sz;
        long off_recv = lsbch_recv * sbchunk_sz;

        if ((rank + 1) % group_size == 0)
            NCCL_CALL(
                ncclSend(d_outbuf + off_send, sbchunk_sz, ncclFloat, lr_send, comm, streams[0])
            );
        else if (rank % group_size == 0) {
            NCCL_CALL(ncclRecv(temp_bufs[0], sbchunk_sz, ncclFloat, lr_recv, comm, streams[0]));

            const int threads = 256;
            long blocks = (sbchunk_sz + threads - 1) / threads;
            add_kernel<<<blocks, threads, 0, streams[0]>>>(
                d_outbuf + off_recv, temp_bufs[0], sbchunk_sz
            );
            CUDA_CALL(cudaGetLastError());
        }

        if ((rank + 1) % group_size == 1)
            NCCL_CALL(
                ncclSend(d_outbuf + off_send, sbchunk_sz, ncclFloat, lr_send, comm, streams[1])
            );
        else if (rank % group_size == 1) {
            NCCL_CALL(ncclRecv(temp_bufs[1], sbchunk_sz, ncclFloat, lr_recv, comm, streams[1]));

            const int threads = 256;
            long blocks = (sbchunk_sz + threads - 1) / threads;
            add_kernel<<<blocks, threads, 0, streams[1]>>>(
                d_outbuf + off_recv, temp_bufs[1], sbchunk_sz
            );
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaStreamSynchronize(streams[1]));
        }
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
            streams[0]
        );
    }

    // --- STEP 5: GLOBAL GATHER ---
    {
        int gch_send = group;
        int gch_recv = (n_ranks - 1 - rank) % n_groups;

        long off_send = gch_send * chunk_sz;
        long off_recv = gch_recv * chunk_sz;

        ncclSendRecv(
            d_outbuf + off_send,
            d_outbuf + off_recv,
            chunk_sz,
            rank,
            gr_send,
            gr_recv,
            comm,
            streams[0]
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
            d_outbuf + off_send,
            d_outbuf + off_recv,
            chunk_sz,
            rank,
            lr_send,
            lr_recv,
            comm,
            streams[0]
        );
    }

    CUDA_CALL(cudaStreamSynchronize(streams[0]));
    for (int i = 0; i < group_size; i++) CUDA_CALL(cudaFree(temp_bufs[i]));
}



// interface function, runs for each rank
void paard_pipelined_nccl(RunArgs* args) {
    long input_size = args->input_size;
    ncclComm_t comm = args->comm;
    int rank, _n_ranks, device;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &_n_ranks);
    ncclCommCuDevice(comm, &device);
    assert(_n_ranks == n_ranks);


    // initialize CUDA stream
    CUDA_CALL(cudaSetDevice(device));
    cudaStream_t streams[group_size];
    for (int i = 0; i < group_size; i++) CUDA_CALL(cudaStreamCreate(&streams[i]));


    // initialize input and output
    float* d_inbuf = nullptr;
    CUDA_CALL(cudaMalloc(&d_inbuf, input_size * sizeof(float)));

    const int threads = 256;
    long blocks = (input_size + threads - 1) / threads;
    init_input_kernel<<<blocks, threads, 0, streams[0]>>>(d_inbuf, rank, input_size);
    CUDA_CALL(cudaGetLastError());

    float* d_outbuf = nullptr;
    CUDA_CALL(cudaMalloc(&d_outbuf, input_size * sizeof(float)));


    // call paard all-reduce
    paard_allreduce(d_inbuf, d_outbuf, input_size, comm, streams);


    // copy back result to host and verify output, short circuit if incorrect
    float* h_res = (float*)malloc(input_size * sizeof(float));
    CUDA_CALL(cudaMemcpy(h_res, d_outbuf, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    *(args->correct) = check_correctness(h_res, rank, n_ranks, input_size, args->atol);
    free(h_res);

    if (!*(args->correct)) {
        CUDA_CALL(cudaFree(d_inbuf));
        CUDA_CALL(cudaFree(d_outbuf));
        for (int i = 0; i < group_size; i++) CUDA_CALL(cudaStreamDestroy(streams[i]));
        return;
    }


    // warmup
    for (int i = 0; i < args->n_warmup; i++)
        paard_allreduce(d_inbuf, d_outbuf, input_size, comm, streams);


    // benchmark
    double* deltas = (double*)malloc(args->n_iters * sizeof(double));
    for (int i = 0; i < args->n_iters; i++) {
        double t0 = get_time();
        paard_allreduce(d_inbuf, d_outbuf, input_size, comm, streams);
        double t1 = get_time();
        deltas[i] = t1 - t0;
    }
    analyze_runtime(args, deltas);
    free(deltas);


    // cleanup
    CUDA_CALL(cudaFree(d_inbuf));
    CUDA_CALL(cudaFree(d_outbuf));
    for (int i = 0; i < group_size; i++) CUDA_CALL(cudaStreamDestroy(streams[i]));
    return;
}
