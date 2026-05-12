# AsyncAllReduce

## How to run

```shell
cd $PSCRATCH/async-ring-allreduce/
./build.sh       # compile, optionally pass -r to build in release mode
sbatch ./run.sh  # run, optionally pass -r to run in release mode, and -n=N_RANKS to run with N_RANKS ranks
```

## Experiment knobs (env vars)

Set before `sbatch` / `srun`; read once per rank after `cudaSetDevice` (`init_benchmark_knob_from_env()` in `src/utils.cu`).

| Variable | Meaning | Default |
|----------|---------|---------|
| `ALLREDUCE_B` | Micro-batches (`b`) for **Pipelined Ring** and **Pipelined HD** only (≥ 2). PAARD ignores this. | `2` |
| `ALLREDUCE_COMPUTE_NS` | `__nanosleep` inside `add_kernel` (simulate reduction cost). | `5000` |
| `ALLREDUCE_INTER_US` | Synthetic delay after each **cross-group** `ncclSend/Recv`: **omit** unset → legacy size-proportional `(float_count >> 8)` ns; **set** to `N` → fixed **`N` microseconds** per step (`N=0` = hardware only). | _(unset,_ legacy prop.) |
| `ALLREDUCE_BENCH_STACK` | Set nonzero (not `0`) to append **`stack_comm_us`** / **`stack_compute_us`** (µs **per iteration, averaged**) for **Classic Ring** and **Pipelined Ring** only. **Same algorithm** (no skipped NCCL steps): CUDA events bracket each `ncclSend`/`ncclRecv` (+ optional synthetic delay) vs each `add_kernel`. Syncs inside brackets **perturb overlap** (pipelined). PAARD / HD leave these at `0`. | _(off)_ |

**Poster stacked bars (measurement-only, one run):**

With `ALLREDUCE_BENCH_STACK=1`, use **Classic Ring** and **Pipelined Ring** rows:

| Quantity | CSV |
|----------|-----|
| Full wall time per iteration | **`avg_latency`** |
| Time in comm brackets | **`stack_comm_us`** |
| Time in `add_kernel` brackets | **`stack_compute_us`** |

**Shares of instrumented time:**  
\(p_{\mathrm{comm}} = \frac{\mathrm{stack\_comm}}{\mathrm{stack\_comm}+\mathrm{stack\_compute}}\),  
\(p_{\mathrm{compute}} = 1 - p_{\mathrm{comm}}\).

**Caveat:** Brackets call **`cudaStreamSynchronize`**, so overlap is perturbed—especially **pipelined** ring. Use **`avg_latency`** for true end-to-end iteration time; use **`stack_*`** for comm vs compute **under this probe**. Compare classic vs pipelined in the **same** instrumented run for a fair split. For best wall times without probe overhead, run again with `ALLREDUCE_BENCH_STACK=0`.

Sweep example:

```shell
ALLREDUCE_BENCH_STACK=1 sbatch ./run_8r.sh -r

# 8-way job (HD + Ring): use ./run_8r.sh (8 GPUs allocated)
ALLREDUCE_B=4 ALLREDUCE_COMPUTE_NS=8000 ALLREDUCE_INTER_US=50 sbatch run.sh -r
```

Older names still work as fallbacks in code only: `ALLREDUCE_N_BATCHES`, `ALLREDUCE_REDUCE_NS`.

## Contributing

To add a new implementation, you will have to modify these files
- `src/your-impl.cu` containing the implementation, refer to `src/interface.h`
- `src/interface.h` containing the function signature for your implementation
- `src/benchmark.cu` with `impls` and `impl_names` updated accordingly
- `bench.sh` to compile with the newly created `your-impl.cu`



ALLREDUCE_B=4 sbatch --output=results/8r_B4.csv --error=results/8r_B4.err ./run_8r.sh -r
 
ALLREDUCE_B=8 sbatch --output=results/8r_B8.csv --error=results/8r_B8.err ./run_8r.sh -r