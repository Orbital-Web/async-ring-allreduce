# AsyncAllReduce

## How to run

```shell
cd $PSCRATCH/async-ring-allreduce/
./build.sh          # compile, optionally pass -r to build in release mode
sbatch ./run.sh     # run, modify node count before running
```

## Implementations

| Name | File | Description |
|------|------|-------------|
| Classic Ring | `src/naive_ringreduce.cu` | Flat ring RS+AG using ncclSend/ncclRecv |
| Pipelined Ring | `src/pipelined_ringreduce_nccl.cu` | 2-stream pipelined RS+AG to overlap compute/comm |
| Hierarchical Ring | `src/hier_ringreduce.cu` | PAARD-style 2-level: intra-node RS (NVLink) → cross-node all-reduce (Slingshot once) → intra-node AG (NVLink) |

## Synthetic inter-node penalty sweep

Since 2 nodes on Perlmutter are likely in the same Dragonfly group (local links only),
we inject an artificial cost on each inter-node hop to model topologies where the
global/local link ratio is larger. The penalty is a LogGP-style affine cost:

```
delay(bytes) = GLOBAL_PENALTY_US us  +  bytes / GLOBAL_BW_GBPS ns
```

- `GLOBAL_PENALTY_US`: fixed per-hop latency (α). Small messages feel this most.
- `GLOBAL_BW_GBPS`: inter-node bandwidth cap (β). Large messages feel this most.

The delay is launched as a single-thread `__nanosleep` kernel on the same CUDA
stream as the NCCL op, so it does **not** block the host and other streams keep
running — this preserves overlap in the pipelined impl. The flat ring pays per
step (14× per all-reduce on 8 ranks); the hierarchical algorithm pays once per
all-reduce, with `bytes` set to the per-rank traffic on the global ring.

```shell
# pure latency sweep
for p in 0 10 100 1000; do GLOBAL_PENALTY_US=$p sbatch run.sh; done

# bandwidth sweep (match Slingshot ~25 GB/s, then squeeze it)
for bw in 25 10 5 1; do GLOBAL_BW_GBPS=$bw sbatch run.sh; done

# combined: 50 us + 10 GB/s link
GLOBAL_PENALTY_US=50 GLOBAL_BW_GBPS=10 sbatch run.sh
```

Results land in `results/bench_<jobid>.csv`. Plot with `utils/plot.py`.

## Contributing

To add a new implementation, you will have to modify these files
- `src/your-impl.cu` containing the implementation, refer to `src/interface.h`
- `src/interface.h` containing the function signature for your implementation
- `src/benchmark.cu` with `impls` and `impl_names` updated accordingly
- `build.sh` to compile with the newly created `your-impl.cu`