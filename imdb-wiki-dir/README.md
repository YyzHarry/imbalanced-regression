# IMDB-WIKI-DIR
## Installation

#### Prerequisites

1. Download and extract IMDB faces and WIKI faces respectively using

```bash
python download_imdb_wiki.py
```

2. __(Optional)__ We have provided required IMDB-WIKI-DIR meta file `imdb_wiki.csv` to set up balanced val/test set in folder `./data`. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_imdb_wiki.py
python data/preprocess_imdb_wiki.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL, wget

## Code Overview

#### Main Files

- `train.py`: main training and evaluation script
- `create_imdb_wiki.py`: create IMDB-WIKI raw meta data
- `preprocess_imdb_wiki.py`: create IMDB-WIKI-DIR meta file `imdb_wiki.csv` with balanced val/test set

#### Main Arguments

- `--data_dir`: data directory to place data and meta file
- `--lds`: LDS switch (whether to enable LDS)
- `--fds`: FDS switch (whether to enable FDS)
- `--reweight`: cost-sensitive re-weighting scheme to use
- `--retrain_fc`: whether to retrain regressor
- `--loss`: training loss type
- `--resume`: path to resume checkpoint (for both training and evaluation)
- `--evaluate`: evaluate only flag
- `--pretrained`: path to load backbone weights for regressor re-training (RRT)

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

#### Train a model with different losses

To use Focal-R loss

```bash
python train.py --loss focal_l1
```

To use huber loss

```bash
python train.py --loss huber
```

#### Train a model using RRT

```bash
python train.py [...retrained model arguments...] --retrain_fc --pretrained <path_to_pretrained_ckpt>
```

#### Train a model using LDS

To use Gaussian kernel (kernel size: 5, sigma: 2)

```bash
python train.py --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2
```

#### Train a model using FDS

To use Gaussian kernel (kernel size: 5, sigma: 2)

```bash
python train.py --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2
```
#### Train a model using LDS + FDS
```bash
python train.py --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2
```

#### Evaluate a trained checkpoint

```bash
python train.py [...evaluation model arguments...] --evaluate --resume <path_to_evaluation_ckpt>
```

## Reproduced Benchmarks and Model Zoo

We provide below reproduced results on IMDB-WIKI-DIR (base method `SQINV`, metric `MAE`).
Note that some models could give **better** results than the reported numbers in the paper.


|   Model   | Overall | Many-Shot | Medium-Shot | Few-Shot | Download |
| :-------: | :-----: | :-------: | :---------: | :------: | :------: |
|    LDS    |  7.87   |   7.31    |   12.45     |   22.60  | [model](https://drive.google.com/file/d/1HnGw1gs6UAlvbol4EulHX_Kqx_pwJZ70/view?usp=sharing) |
|    FDS    |  7.66   |   7.06    |   12.60     |   22.37  | [model](https://drive.google.com/file/d/1H7_dDMn83-paFrcrEmOiDLZmoham4js9/view?usp=sharing) |
| LDS + FDS |  7.68   |   7.07    |   12.79     |   21.85  | [model](https://drive.google.com/file/d/1C_YxpTW-rhCRIF4wnFShojp5ydAjFmHo/view?usp=sharing) |