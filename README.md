
# A Generalized Approach to Redundant Calibration with JAX

The Hydrogen Epoch of Reionization Array (HERA) relies on redundant calibration with [redcal](https://github.com/HERA-Team/hera_cal/blob/master/hera_cal/redcal.py) from the [hera_cal](https://github.com/HERA-Team/hera_cal) package to calibrate its data, which assumes Gaussian noise statistics, linearizes the measurement equation and minimizes the χ2. 

In this repository, we show generalization of this maximum likelihood estimation (MLE) to non-Gaussian statistics and without the need for linearization, which can be achieved at good computational performance with very little programming effort with the [JAX](https://github.com/google/jax) machine learning library. 

As example, we works with and compare Gaussian and Cauchy assumed noise distributions in the calibration of HERA datasets, with the latter showing expected resilience to radio-frequency interference.

Check out the [SimpleRedCal.ipynb](SimpleRedCal.ipynb) notebook for full example calibrations.
