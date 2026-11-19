#!/usr/bin/env bash
# Run neurodamus simulations one after another, one per config folder.
#
# Usage:
#   ./run_simulations.sh
#
# Optional environment overrides:
#   NP=64 ./run_simulations.sh                          # change number of MPI ranks
#   MPI_EXTRA_ARGS="--allow-run-as-root" ./run_simulations.sh   # extra mpirun flags
#   STOP_ON_ERROR=1 ./run_simulations.sh                # stop at the first failed run

set -u

# ---------------------------------------------------------------------------
# EDIT THIS: the folders to run, in order. Each must contain simulation_config.json
# ---------------------------------------------------------------------------
FOLDERS=(
    "/sim15Hz0.5Vm_seed2"
    "/sim15Hz0.5Vm_seed3"
    "/sim10Hz0.5Vm_seed2"
    "/sim10Hz0.5Vm_seed3"
)

# Directory that contains ./x86_64/special (defaults to where you launch the script)
WORKDIR="${WORKDIR:-$(pwd)}"

NP="${NP:-120}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
LOG_DIR="${LOG_DIR:-$WORKDIR/sim_logs}"

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
if [[ -z "${NEURODAMUS_PYTHON:-}" ]]; then
    echo "ERROR: NEURODAMUS_PYTHON is not set." >&2
    exit 1
fi

cd "$WORKDIR" || { echo "ERROR: cannot cd to $WORKDIR" >&2; exit 1; }

if [[ ! -x ./x86_64/special ]]; then
    echo "ERROR: ./x86_64/special not found or not executable in $WORKDIR" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
succeeded=()
failed=()
total=${#FOLDERS[@]}
i=0

for folder in "${FOLDERS[@]}"; do
    i=$((i + 1))
    folder="${folder%/}"
    config="$folder/sim_with_stim.json"
    name="$(basename "$folder")"
    log="$LOG_DIR/${i}_${name}.log"

    echo "=================================================================="
    echo "[$i/$total] $(date '+%Y-%m-%d %H:%M:%S')  Running: $folder"
    echo "  Log: $log"

    if [[ ! -f "$config" ]]; then
        echo "  SKIPPED: $config does not exist"
        failed+=("$folder (missing config)")
        [[ "$STOP_ON_ERROR" == "1" ]] && break
        continue
    fi

    start=$(date +%s)

    # shellcheck disable=SC2086
    mpirun -np "$NP" $MPI_EXTRA_ARGS \
        ./x86_64/special -mpi \
        -python "$NEURODAMUS_PYTHON/init.py" \
        --configFile="$config" \
        2>&1 | tee "$log"
    status=${PIPESTATUS[0]}

    elapsed=$(( $(date +%s) - start ))
    printf -v duration '%02dh:%02dm:%02ds' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))

    if [[ $status -eq 0 ]]; then
        echo "  DONE in $duration"
        succeeded+=("$folder")
    else
        echo "  FAILED (exit code $status) after $duration"
        failed+=("$folder (exit $status)")
        [[ "$STOP_ON_ERROR" == "1" ]] && break
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=================================================================="
echo "Finished $(date '+%Y-%m-%d %H:%M:%S')"
echo "Succeeded: ${#succeeded[@]}   Failed: ${#failed[@]}   Total: $total"
for f in "${failed[@]}"; do
    echo "  FAILED: $f"
done

[[ ${#failed[@]} -eq 0 ]]