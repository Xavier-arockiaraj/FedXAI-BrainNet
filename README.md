# FedXAI-BrainNet: code

| File | Paper section | Language |
|---|---|---|
| `fedxai_brainnet_simulation.py` | Section 6 simulation study (Figs. 3-7, Table 4) | Python 3 (numpy, scipy, scikit-learn, matplotlib) |
| `fetal_planes_db_mobilenetv2_corrected.m` | Section 6.7 corrected real-data experiment on FETAL_PLANES_DB | MATLAB (Deep Learning Toolbox + MobileNet-v2 support package) |
| `results/` | Sample output of the Python script (seed 42) | |

## Simulation study
    pip install numpy scipy scikit-learn matplotlib
    python fedxai_brainnet_simulation.py --out results
Options: `--seed`, `--rounds` (6), `--local-epochs` (40), `--ensemble` (15),
`--dp-sigma` / `--dp-clip` (Gaussian DP on client updates, Eq. 2; off by default),
`--contrast` / `--jitter` (task difficulty). Runtime is about 30 s on a laptop CPU.

Module-to-equation map: FedAvg = Eq. 1, DP clipping/noise = Eq. 2,
ensemble mean/variance = Eqs. 3-4, occlusion deletion-AUC = proxy for Eqs. 5-6,
INT8 quantization = Eq. 7.

## Real-data experiment
1. Download and unzip https://doi.org/10.5281/zenodo.3904280
2. Set `DATA_ROOT` at the top of the .m file and run it.
Labels come from the brain-plane column of the metadata file (not folder names),
the dataset's patient-disjoint train/test split is used, and test accuracy is
reported per ultrasound machine. Set `RUN_LOMO = true` for leave-one-machine-out.
