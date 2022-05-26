# Constrained GPI (CGPI)

This is part of our submission to NeurIPS 2022, "Constrained GPI for Zero-Shot Transfer in Reinforcement Learning".

It includes the implementation for universal successor features approximators (USFAs) and our test-time approach, constrained GPI (CGPI).

### Requirements
This code is tested in environments with the following conditions:
* Ubuntu 16.04 machine
* Python 3.7.11
* [MuJoCo](http://mujoco.org/) 

### Environment Setup
```
conda env create -f environment.yml
```

### Training
For the training of USFAs in Reacher, within the created conda environment `psfs`, run commands like the following:
```
python test_scripts/train_reacher_usfa.py --run_group usfas --seed 1
```

It will create experiment directories under `groups/usfas`, which also contain model checkpoint files.

### Evaluation
Using Jupyter, open `ipynbs/reacher_eval.ipynb` and input the model checkpoint paths into the variable `psi_ckpts_infos_test`.  
Then, you can run the notebook file to get the plots generated with [Rliable](https://github.com/google-research/rliable/).

