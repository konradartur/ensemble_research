# ensemble_research

## Assignment project for Machine Learning course at Jagiellonian University

Mentors: @kudkudak, @mchuck

## Contents

See `report.pdf` file. To reproduce the results run command:

```python main.py reproduce experiment_name```

where `experiment_name` is `delayed_ensemble` or `noisy_labels` (or unreported `regularizers`).
It will train the network (using CUDA if possible), pickle predictions and make plots (or just print output to console in case of `regularizers`).

CIFAR10-C has to be downloaded from the website linked in dataset file it is defined in.
