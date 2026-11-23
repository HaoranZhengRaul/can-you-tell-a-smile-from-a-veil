#!/bin/bash
# Session 8 generic SLURM wrapper: sbatch [opts] code/job.sh <command ...>
# Sets TMPDIR on /scratch (never /tmp), runs from the project root, records the command.
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --output=/project/dmv_project/projects/ai_pricing_kbb/log/%x_%j.out
#SBATCH --error=/project/dmv_project/projects/ai_pricing_kbb/err/%x_%j.err
set -euo pipefail
# TMPDIR: /scratch/$USER/$JOBID, or /dev/shm when the shared /scratch is full (both allowed; never /tmp)
export TMPDIR="/scratch/$USER/$SLURM_JOB_ID"
if ! mkdir -p "$TMPDIR" 2>/dev/null; then
  export TMPDIR="/dev/shm/${USER}_$SLURM_JOB_ID"; mkdir -p "$TMPDIR"
  echo "note: /scratch not writable; TMPDIR=$TMPDIR"
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /project/dmv_project/projects/ai_pricing_kbb
echo "host $(hostname) start $(date -u +%FT%TZ) cmd: $*"
"$@"
echo "end $(date -u +%FT%TZ)"
rm -rf "$TMPDIR"
