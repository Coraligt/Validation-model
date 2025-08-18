#!/bin/bash
#SBATCH --job-name=git-newvalid
#SBATCH --account=gts-akhan40
#SBATCH --partition=gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=git-newvalid-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cli872@gatech.edu


# Load required modules
module load cuda/12.1.0
module load anaconda3

set -euo pipefail

REPO="/storage/coda1/p-akhan40/0/cli872/Validation-model"
cd "$REPO"

rm -f .git/index.lock .git/packed-refs.lock .git/shallow.lock || true
git merge --abort 2>/dev/null || true
git rebase --abort 2>/dev/null || true

# Fetch new branch
git fetch --depth=1 origin newvalid --progress

if ! git rev-parse --verify newvalid >/dev/null 2>&1; then
  git checkout -b newvalid origin/newvalid
else
  git checkout newvalid
  git reset --hard origin/newvalid
fi
