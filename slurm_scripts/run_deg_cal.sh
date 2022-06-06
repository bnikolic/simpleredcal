#!/bin/sh

# Set SBATCH Directives
# Lines starting with "#SBATCH", before any shell commands are
# interpreted as command line arguments to sbatch.
# Don't put any commands before the #SBATCH directives or they will not work.
#
#SBATCH --export=ALL                                       # Export all environment variables to job
#SBATCH --partition=hera                                   # Specify partition on which to run job
#SBATCH --mem=64G                                          # Amount of memory needed by the whole job
#SBATCH -D /lustre/aoc/projects/hera/mmolnar/simpleredcal  # Working directory
#SBATCH --mail-type=BEGIN,END,FAIL                         # Send email on begin, end, and fail of job
#SBATCH --nodes=1                                          # Request 1 node
#SBATCH --ntasks-per-node=4                                # Request 4 cores
#SBATCH --time=12:00:00                                    # Request 12 hours, 0 minutes and 0 seconds.
#SBATCH --output=deg_cal.2458098.43869.ee.jd.2458099.gaussian.out

source ~/.bashrc

conda activate hera

echo "start: $(date)"

cd /lustre/aoc/projects/hera/mmolnar/simpleredcal

python script/deg_cal.py '2458098.43869' --deg_dim 'jd' \
--pol 'ee' --dist 'gaussian' --coords 'cartesian' --tgt_jd 2458099 \
--rel_dir 'rel_dfs' --out_dir 'deg_dfs'

echo "end: $(date)"
