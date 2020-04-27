#!/bin/sh

# Don't put any commands before the #PBS options or they will not work
#PBS -V # Export all environment variables from the qsub commands environment to the batch job.
#PBS -l pmem=64gb,pvmem=64gb # Amount of memory needed by each processor (ppn) in the job.
#PBS -d /lustre/aoc/projects/hera/mmolnar/simpleredcal # Working directory (PBS_O_WORKDIR) set to your Lustre area
#PBS -m ae # Send email when Jobs end or abort

# casa's python requires a DISPLAY for matplot, so create a virtual X server
/lustre/aoc/projects/hera/mmolnar/anaconda3/envs/hera/bin/python rel_cal.py 2458098.43869 --pol 'ee'