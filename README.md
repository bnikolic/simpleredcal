
# A Generalized Approach to Redundant Calibration with JAX

The Hydrogen Epoch of Reionization Array (HERA) relies on redundant calibration with [redcal](https://github.com/HERA-Team/hera_cal/blob/master/hera_cal/redcal.py) from the [hera_cal](https://github.com/HERA-Team/hera_cal) package to calibrate its data, which assumes Gaussian noise statistics, linearizes the measurement equation and minimizes the χ2. 

In this repository, we show generalization of this maximum likelihood estimation (MLE) to non-Gaussian statistics and without the need for linearization, which can be achieved at good computational performance with very little programming effort with the [JAX](https://github.com/google/jax) machine learning library. 

As an example, we works with and compare Gaussian and Cauchy assumed noise distributions in the calibration of HERA datasets, with the latter showing expected resilience to radio-frequency interference.

We also provide tools to compare redundantly calibrated visibilities across multiple days, which may have dissimilar degenerate parameters. We also extend redundant calibration to find the best estimates of the location and scale parameters for visibilities across multiple days.

Check out the [SimpleRedCal.ipynb](https://nbviewer.jupyter.org/github/bnikolic/simpleredcal/blob/master/SimpleRedCal.ipynb) notebook for full example calibrations.

### Associated publications:
 - [HERA Memorandum #84: A Generalized Approach to Redundant Calibration with JAX](http://reionization.org/wp-content/uploads/2013/03/HERA084__A_Generalized_Approach_to_Redundant_Calibration_with_JAX.pdf)
 - [HERA Memorandum #94: Comparing Visibility Solutions from Relative Redundant Calibration by Degenerate Translation](http://reionization.org/manual_uploads/HERA094__Comparing_Visibility_Solutions_from_Relative_Redundant_Calibration_by_Degenerate_Translation.pdf)

Note that the above memoranda use code from [this](https://github.com/bnikolic/simpleredcal/tree/58c954e4e5fcb6e9b81fb7d523c5eb97a24ffb60) commit. Since then, the repository has been packaged; update memoranda can be found [here](https://github.com/bnikolic/simpleredcal/tree/master/publications).
