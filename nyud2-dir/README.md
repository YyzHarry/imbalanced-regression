# NYUD2-DIR
## Installation

#### Prerequisites

1. Download and extract NYU v2 dataset to folder `./data` using

```bash
python download_nyud2.py
```

2. __(Optional)__ We have provided required meta files `nyu2_train_FDS_subset.csv` and `test_balanced_mask.npy`  for efficient FDS feature statistics computation and balanced test set mask in folder `./data`. To reproduce the results in the paper, please directly use these two files. If you want to try different FDS computation subsets and balanced test set masks, you can run

```bash
python preprocess_nyud2.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- numpy, pandas, scipy, tqdm, matplotlib, PIL, gdown, tensorboardX

## Code Overview

#### Main Files

- `train.py`: main training script
- `test.py`: main evaluation script
- `preprocess_nyud2.py`: create meta files `nyu2_train_FDS_subset.csv` and `test_balanced_mask.npy` for NYUD2-DIR

#### Main Arguments

For `train.py`:

- `--data_dir`: data directory to place data and meta file
- `--lds`: LDS switch (whether to enable LDS)
- `--fds`: FDS switch (whether to enable FDS)
- `--reweight`: cost-sensitive re-weighting scheme to use
- `--resume`: whether to resume training (only for training)
- `--retrain_fc`: whether to retrain regressor
- `--pretrained`: path to load backbone weights for regressor re-training (RRT)

For `test.py`:

- `--eval_model`: path to resume checkpoint (only for evaluation)

## Getting Started

#### Train a vanilla model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --data_dir <path_to_data_dir> --reweight none
```

Always specify `CUDA_VISIBLE_DEVICES` for GPU IDs to be used (by default, 4 GPUs) and `--data_dir` when training a model or directly fix your default data directory path in the code. We will omit these arguments in the following for simplicity.


#### Train a model using re-weighting

To perform inverse re-weighting

```bash
python train.py --reweight inverse
```

To perform square-root inverse re-weighting

```bash
python train.py --reweight sqrt_inv
```

#### Train a model using RRT

```bash
python train.py [...retrained model arguments...] --retrain_fc --pretrained <path_to_pretrained_ckpt>
```

#### Train a model using LDS

To use Gaussian kernel (kernel size: 5, sigma: 2)

```bash
python train.py --reweight inverse --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2
```

#### Train a model using FDS

To use Gaussian kernel (kernel size: 5, sigma: 2)

```bash
python train.py --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2
```

#### Train a model using LDS + FDS

```bash
python train.py --reweight inverse --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2
```

#### Evaluate a trained checkpoint

```bash
python test.py --data_dir <path_to_data_dir> --eval_model <path_to_evaluation_ckpt>
```

## Reproduced Benchmarks and Model Zoo

We provide below reproduced results on NYUD2-DIR (base method `Vanilla`, metric `RMSE`).
Note that some models could give **better** results than the reported numbers in the paper.

|   Model   | Overall | Many-Shot | Medium-Shot | Few-Shot | Download |
| :-------: | :-----: | :-------: | :---------: | :------: | :------: |
|    LDS    |  1.387  |   0.671   |    0.913    |  1.954   | [model](https://drive.google.com/file/d/1RgQx-nreiJ-chH0887xCy7gxah-zrrEO/view?usp=sharing) |
|    FDS    |  1.442  |   0.615   |    0.940    |  2.059   | [model](https://drive.google.com/file/d/1FEKzBzMPaGubmv9iK4BP6LJng44Mhc7s/view?usp=sharing) |
| LDS + FDS |  1.301  |   0.731   |    0.832    |  1.799   | [model](https://drive.google.com/file/d/1QlZJOPYSyRRFqa1Q-y7-JlTDABiQZJUF/view?usp=sharing) |