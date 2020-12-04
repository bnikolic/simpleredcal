#!/bin/sh

# Set PBS Directives
# Lines starting with "#PBS", before any shell commands are
# interpreted as command line arguments to qsub.
# Don't put any commands before the #PBS options or they will not work
#
#PBS -V # Export all environment variables from the qsub commands environment to the batch job.
#PBS -l mem=64gb,pvmem=96gb # Amount of memory needed by each processor (ppn) in the job.
#PBS -d /lustre/aoc/projects/hera/mmolnar/simpleredcal # Working directory (PBS_O_WORKDIR) set to your Lustre area
#PBS -m bea # Send email when Jobs end or abort
#PBS -l nodes=1:ppn=2 # default is 1 core on 1 node
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o rel_cal.2458098.43869.ee.gaussian.out

/lustre/aoc/projects/hera/mmolnar/anaconda3/envs/hera/bin/python rel_cal.py 2458098.43869 --pol 'ee' --flag_type 'first' \
--dist 'gaussian' --method 'polar' --logamp --tilt_reg --gphase_reg
