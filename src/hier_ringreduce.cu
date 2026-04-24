// hier_ringreduce.cu
// 2-level hierarchical all-reduce (PAARD-style) using ncclCommSplit.
//
// Algorithm:
//   1. Intra-node reduce-scatter via ncclReduceScatter on local_comm (fast NVLink).
//   2. Cross-node all-reduce of each rank's scattered chunk via ncclAllReduce on
//      global_comm (one Slingshot crossing per all-reduce).
//   3. Intra-node all-gather via ncclAllGather on local_comm (fast NVLink).
//
// With n_nodes=2, n_local=4, the flat ring crosses the inter-node fabric
// 2*(n_ranks-1) = 14 times; the hierarchical algorithm crosses it once.
//
// Requires NCCL >= 2.14 (ncclCommSplit).
// input_size must be divisible by local_size.

#include <assert.h>
#include <stdio.h>

#include "interface.h"



// perform one hierarchical all-reduce
static void hier_allreduce(
    const float* d_inbuf,
    float* d_outbuf,
    long input_size,
    ncclComm_t local_comm,  // intra-node communicator
    ncclComm_t global_comm, // inter-node communicator (same local_rank across nodes)
    cudaStream_t stream
) {
    int local_rank, local_size;
    ncclCommUserRank(local_comm, &local_rank);
    ncclCommCount(local_comm, &local_size);

    assert(input_size % local_size == 0);
    long chunk_size = input_size / local_size;

    // this rank's chunk within the output buffer (in-place RS/AG)
    float* my_chunk = d_outbuf + (long)local_rank * chunk_size;

    if (d_inbuf != d_outbuf)
        CUDA_CALL(cudaMemcpyAsync(
            d_outbuf, d_inbuf, input_size * sizeof(float), cudaMemcpyDeviceToDevice, stream
        ));

    // Step 1: intra-node reduce-scatter
    // after this, my_chunk holds the partial sum from all ranks on this node
    NCCL_CALL(ncclReduceScatter(
        d_outbuf, my_chunk, chunk_size, ncclFloat, ncclSum, local_comm, stream
    ));

    // inject inter-node link penalty for the entire global all-reduce phase:
    // each rank's ring-allreduce over global_comm moves
    //   2 * (n_nodes - 1) / n_nodes  *  chunk_size  floats
    // across the slow link. we fold all of that traffic into one delay call.
    int n_nodes;
    ncclCommCount(global_comm, &n_nodes);
    long global_bytes = (n_nodes > 1)
        ? (long)(2 * (n_nodes - 1)) * chunk_size * (long)sizeof(float) / n_nodes
        : 0;
    maybe_penalize_internode(stream, global_bytes);

    // Step 2: cross-node all-reduce — each rank exchanges its chunk with the
    // corresponding rank on every other node; my_chunk becomes the global sum
    NCCL_CALL(ncclAllReduce(
        my_chunk, my_chunk, chunk_size, ncclFloat, ncclSum, global_comm, stream
    ));

    // Step 3: intra-node all-gather — broadcast fully-reduced chunk to all local ranks
    NCCL_CALL(ncclAllGather(my_chunk, d_outbuf, chunk_size, ncclFloat, local_comm, stream));

    CUDA_CALL(cudaStreamSynchronize(stream));
}



// interface function, runs for each rank
void ring_hierarchical(RunArgs* args) {
    long input_size = args->input_size;
    ncclComm_t comm = args->comm;
    int node_id = args->node_id;
    int rank, n_ranks, device;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &n_ranks);
    ncclCommCuDevice(comm, &device);

    CUDA_CALL(cudaSetDevice(device));

    // build local_comm: all ranks on the same node share the same color (node_id)
    ncclComm_t local_comm;
    NCCL_CALL(ncclCommSplit(comm, node_id, rank, &local_comm, NULL));

    int local_rank, local_size;
    ncclCommUserRank(local_comm, &local_rank);
    ncclCommCount(local_comm, &local_size);

    // build global_comm: ranks with the same local position across nodes share color
    // ranks with local_rank=0 form one global_comm, local_rank=1 form another, etc.
    ncclComm_t global_comm;
    NCCL_CALL(ncclCommSplit(comm, local_rank, rank, &global_comm, NULL));

    // guard: input must be evenly partitioned across local ranks
    if (input_size % local_size != 0) {
        if (rank == 0)
            fprintf(
                stderr,
                "hier: input_size %ld not divisible by local_size %d, skipping\n",
                input_size,
                local_size
            );
        *(args->correct) = false;
        NCCL_CALL(ncclCommDestroy(local_comm));
        NCCL_CALL(ncclCommDestroy(global_comm));
        return;
    }


    // initialize CUDA stream
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


    // correctness check
    hier_allreduce(d_inbuf, d_outbuf, input_size, local_comm, global_comm, stream);

    float* h_res = (float*)malloc(input_size * sizeof(float));
    CUDA_CALL(cudaMemcpy(h_res, d_outbuf, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    *(args->correct) = check_correctness(h_res, rank, n_ranks, input_size, args->atol);
    free(h_res);

    if (!*(args->correct)) {
        CUDA_CALL(cudaFree(d_inbuf));
        CUDA_CALL(cudaFree(d_outbuf));
        CUDA_CALL(cudaStreamDestroy(stream));
        NCCL_CALL(ncclCommDestroy(local_comm));
        NCCL_CALL(ncclCommDestroy(global_comm));
        return;
    }


    // warmup
    for (int i = 0; i < args->n_warmup; i++)
        hier_allreduce(d_inbuf, d_outbuf, input_size, local_comm, global_comm, stream);


    // benchmark
    double* deltas = (double*)malloc(args->n_iters * sizeof(double));
    for (int i = 0; i < args->n_iters; i++) {
        double t0 = get_time();
        hier_allreduce(d_inbuf, d_outbuf, input_size, local_comm, global_comm, stream);
        double t1 = get_time();
        deltas[i] = t1 - t0;
    }
    analyze_runtime(args, deltas);
    free(deltas);


    // cleanup
    CUDA_CALL(cudaFree(d_inbuf));
    CUDA_CALL(cudaFree(d_outbuf));
    CUDA_CALL(cudaStreamDestroy(stream));
    NCCL_CALL(ncclCommDestroy(local_comm));
    NCCL_CALL(ncclCommDestroy(global_comm));
}
