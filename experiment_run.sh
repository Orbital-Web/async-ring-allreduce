#!/bin/bash
# experiment_run.sh
# submits the full inter-node penalty sweep as separate sbatch jobs.
# each job writes directly to a named results/*.csv — no manual renaming needed.
#
# usage:
#   ./experiment_run.sh                 # submit all sweep points (9 jobs)
#   ./experiment_run.sh --baseline      # just the zero-penalty baseline
#   ./experiment_run.sh --latency       # just the latency sweep (3 jobs)
#   ./experiment_run.sh --bandwidth     # just the bandwidth sweep (4 jobs)
#   ./experiment_run.sh --combined      # just the combined point (1 job)
#   ./experiment_run.sh --force         # resubmit even if CSV exists
#   ./experiment_run.sh --dry-run       # print commands without submitting
#
# safe to re-run: jobs whose CSV already exists are skipped.
# if the qos submit limit is hit, the script sleeps and retries every 60s.
#
# after all jobs finish:
#   squeue -u $USER                              # wait until empty
#   ls results/experiment_run/8gpu_*.csv         # inspect outputs

set -uo pipefail

RESULTS_DIR="results/experiment_run"
mkdir -p "$RESULTS_DIR"

DRY_RUN=0
RUN_BASELINE=1
RUN_LATENCY=1
RUN_BANDWIDTH=1
RUN_COMBINED=1
SKIP_EXISTING=1       # if CSV already exists, skip (re)submission
POLL_INTERVAL=60      # seconds between retries when QoS submit-limit is hit

if [[ $# -gt 0 ]]; then
    RUN_BASELINE=0
    RUN_LATENCY=0
    RUN_BANDWIDTH=0
    RUN_COMBINED=0
    for arg in "$@"; do
        case $arg in
            --baseline)  RUN_BASELINE=1 ;;
            --latency)   RUN_LATENCY=1 ;;
            --bandwidth) RUN_BANDWIDTH=1 ;;
            --combined)  RUN_COMBINED=1 ;;
            --force)     SKIP_EXISTING=0 ;;
            --dry-run)
                DRY_RUN=1
                RUN_BASELINE=1
                RUN_LATENCY=1
                RUN_BANDWIDTH=1
                RUN_COMBINED=1
                ;;
            *)
                echo "unknown flag: $arg"
                exit 1
                ;;
        esac
    done
fi

# submit one sweep point, retrying when the QoS submit-limit is hit.
#   $1 = output basename (no extension), e.g. 8gpu_p100
#   $2 = env-var assignments as a single string, e.g. "GLOBAL_PENALTY_US=100"
submit() {
    local name="$1"
    local envs="$2"
    local out="$RESULTS_DIR/${name}.csv"
    local err="$RESULTS_DIR/${name}.err"

    # skip if the CSV already exists (resumable re-runs)
    if [[ $SKIP_EXISTING -eq 1 && -s "$out" ]]; then
        echo "  [$name]  skip — $out already exists (use --force to resubmit)"
        return 0
    fi

    local cmd="$envs sbatch --output=$out --error=$err run.sh -r"
    echo "  [$name]  $cmd"

    if [[ $DRY_RUN -eq 1 ]]; then
        return 0
    fi

    # keep retrying on QoS submit-limit rejections; other failures bubble up
    while :; do
        local sbatch_out
        sbatch_out=$(eval "$cmd" 2>&1)
        local rc=$?
        echo "    $sbatch_out"
        if [[ $rc -eq 0 ]]; then
            return 0
        fi
        if [[ "$sbatch_out" == *"QOSMaxSubmitJobPerUserLimit"* ]]; then
            echo "    -> queue full, sleeping ${POLL_INTERVAL}s and retrying..."
            sleep "$POLL_INTERVAL"
            continue
        fi
        echo "    sbatch failed with exit code $rc (not a QoS limit) — aborting"
        return $rc
    done
}

echo "=== async-ring-allreduce penalty sweep ==="
[[ $DRY_RUN -eq 1 ]] && echo "(dry-run — no jobs will be submitted)"
echo

if [[ $RUN_BASELINE -eq 1 ]]; then
    echo "-- baseline (no penalty) --"
    submit "8gpu_baseline"  ""
    echo
fi

if [[ $RUN_LATENCY -eq 1 ]]; then
    echo "-- latency sweep (GLOBAL_PENALTY_US) --"
    for p in 10 100 1000; do
        submit "8gpu_p${p}"  "GLOBAL_PENALTY_US=${p}"
    done
    echo
fi

if [[ $RUN_BANDWIDTH -eq 1 ]]; then
    echo "-- bandwidth sweep (GLOBAL_BW_GBPS) --"
    for bw in 25 10 5 1; do
        submit "8gpu_bw${bw}"  "GLOBAL_BW_GBPS=${bw}"
    done
    echo
fi

if [[ $RUN_COMBINED -eq 1 ]]; then
    echo "-- combined point (latency + bandwidth) --"
    submit "8gpu_p50_bw10"  "GLOBAL_PENALTY_US=50 GLOBAL_BW_GBPS=10"
    echo
fi

if [[ $DRY_RUN -eq 0 ]]; then
    echo "=== submitted. track progress with: ==="
    echo "    squeue -u \$USER"
    echo "    ls -lt $RESULTS_DIR/8gpu_*.csv"
fi
