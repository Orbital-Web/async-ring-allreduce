# AsyncAllReduce

## How to run

```shell
cd $PSCRATCH/async-ring-allreduce/
./build.sh       # compile, optionally pass -r to build in release mode
sbatch ./run.sh  # run, optionally pass -r to run in release mode, and -n=N_RANKS to run with N_RANKS ranks
```

## Experiment knobs (4 env vars)

Set before `sbatch` / `srun`; read once per rank after `cudaSetDevice` (`init_benchmark_knob_from_env()` in `src/utils.cu`).

| Variable | Meaning | Default |
|----------|---------|---------|
| `ALLREDUCE_B` | Micro-batches (`b`) for **Pipelined Ring** and **Pipelined HD** only (≥ 2). PAARD ignores this. | `2` |
| `ALLREDUCE_COMPUTE_NS` | `__nanosleep` inside `add_kernel` (simulate reduction cost). | `5000` |
| `ALLREDUCE_INTER_US` | Synthetic delay after each **cross-group** `ncclSend/Recv`: **omit** unset → legacy size-proportional `(float_count >> 8)` ns; **set** to `N` → fixed **`N` microseconds** per step (`N=0` = hardware only). | _(unset,_ legacy prop.) |
| `ALLREDUCE_BENCH_STACK` | Set nonzero (not `0`) to append **`stack_comm_us`** / **`stack_compute_us`** CSV columns (**Classic Ring** / **Pipelined Ring** only). Uses CUDA events + per-step sync; totals are for **poster-style comm vs compute breakdown**, **not** true overlapped time for pipelined runs. PAARD / HD leave these columns zero until instrumented. | _(off)_ |

**Stacked-bar timing:** Instrumentation records intervals around grouped `ncclSend`/`Recv` (+ optional `sim_latency_kernel`) as communication and around `add_kernel` as computation. Syncing inside each bracket **serializes** the schedule versus the timed wall-clock loop, especially for pipelined overlap—compare **full** line vs **summed stacks** accordingly.

Sweep example:

```shell
ALLREDUCE_B=4 ALLREDUCE_COMPUTE_NS=8000 ALLREDUCE_INTER_US=50 sbatch run.sh -r

# Posters — comm vs compute columns (Classic / Pipelined Ring only): 
ALLREDUCE_BENCH_STACK=1 sbatch ./run_8r.sh -r

# 8-way job (HD + Ring): use ./run_8r.sh (8 GPUs allocated)
sbatch ./run_8r.sh -r
```

Older names still work as fallbacks in code only: `ALLREDUCE_N_BATCHES`, `ALLREDUCE_REDUCE_NS`.

## Contributing

To add a new implementation, you will have to modify these files
- `src/your-impl.cu` containing the implementation, refer to `src/interface.h`
- `src/interface.h` containing the function signature for your implementation
- `src/benchmark.cu` with `impls` and `impl_names` updated accordingly
- `bench.sh` to compile with the newly created `your-impl.cu`