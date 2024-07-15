# pident2
Source code for "Using attention neural networks to estimate individual-level pig growth trajectories from group-level weight time series".

Packages used:
- Python v3.8.16
- Pytorch v1.13.1
- Cuda v11.7.1
- Captum v0.6.0
- Numpy v1.22.3
- Scipy v1.7.3
- Scikit-learn v1.1.1
- Matplotlib v3.2.2
- Seaborn v0.12.2

Script list:
- train.py - To train a new model.
- metrics.py - To evaluate a model's predictive performance. Also used to generate the necessary files for growth forecasting with the "--index" option.
- forecast.py - To use predicted growth trajectories in a downstream growth forecasting task.
- interpret.py - To interpret predictions made by a model.
