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
#PBS -l nodes=1:ppn=4 # default is 1 core on 1 node
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o deg_cal.2458098.43869.ee.jd.2458099.gaussian.out

/users/mmolnar/anaconda3/envs/hera/bin/python deg_cal.py '2458098.43869' --deg_dim 'jd' \
--pol 'ee' --dist 'gaussian' --coords 'cartesian'
