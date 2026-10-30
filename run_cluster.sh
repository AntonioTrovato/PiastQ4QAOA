#!/bin/bash
# Launches the whole study on the PCSS cluster through Slurm:
#     sbatch run_cluster.sh            # fresh run
#     sbatch run_cluster.sh --resume   # continue an interrupted run
# The real-machine switches are set HERE, not in the repo files, so the repo
# stays on the safe offline simulator and nothing real starts by accident.
# One CPU-light job: it mostly waits for the QPU queue; the partition has no
# time limit. Token: ~/.piastq_token (chmod 600, never in git).
#SBATCH --job-name=piastq
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm_%j.out

cd "$HOME/PiastQ4QAOA" || exit 1
mkdir -p logs
source qiskit_env/bin/activate

export PIASTQ_API_TOKEN="$(cat "$HOME/.piastq_token")"
export PIASTQ_BACKEND_NAME=piast_q
export PIASTQ_ALLOW_REAL_HARDWARE=1

python run_all.py "$@"
