import os
import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from nyu_transform import *
from scipy.ndimage import convolve1d
from util import get_lds_kernel_window

# for data loading efficiency
TRAIN_BUCKET_NUM = [0, 0, 0, 0, 0, 0, 0, 25848691, 24732940, 53324326, 69112955, 54455432, 95637682, 71403954, 117244217,
                     84813007, 126524456, 84486706, 133130272, 95464874, 146051415, 146133612, 96561379, 138366677, 89680276,
                     127689043, 81608990, 119121178, 74360607, 106839384, 97595765, 66718296, 90661239, 53103021, 83340912,
                     51365604, 71262770, 42243737, 65860580, 38415940, 53647559, 54038467, 28335524, 41485143, 32106001,
                     35936734, 23966211, 32018765, 19297203, 31503743, 21681574, 16363187, 25743420, 12769509, 17675327,
                     13147819, 15798560, 9547180, 14933200, 9663019, 12887283, 11803562, 7656609, 11515700, 7756306, 9046228,
                     5114894, 8653419, 6859433, 8001904, 6430700, 3305839, 6318461, 3486268, 5621065, 4030498, 3839488, 3220208,
                     4483027, 2555777, 4685983, 3145082, 2951048, 2762369, 2367581, 2546089, 2343867, 2481579, 1722140, 3018892,
                     2325197, 1952354, 2047038, 1858707, 2052729, 1348558, 2487278, 1314198, 3338550, 1132666]

class depthDataset(Dataset):
    def __init__(self, data_dir, csv_file, mask_file=None, transform=None, args=None):
        self.data_dir = data_dir
        self.frame = pd.read_csv(csv_file, header=None)
        self.mask = torch.tensor(np.load(mask_file), dtype=torch.bool) if mask_file is not None else None
        self.transform = transform
        self.bucket_weights = self._get_bucket_weights(args) if args is not None else None

    def _get_bucket_weights(self, args):
        assert args.reweight in {'none', 'inverse', 'sqrt_inv'}
        assert args.reweight != 'none' if args.lds else True, "Set reweight to \'sqrt_inv\' or \'inverse\' (default) when using LDS"
        if args.reweight == 'none':
            return None
        logging.info(f"Using re-weighting: [{args.reweight.upper()}]")

        if args.lds:
            value_lst = TRAIN_BUCKET_NUM[args.bucket_start:]
            lds_kernel_window = get_lds_kernel_window(args.lds_kernel, args.lds_ks, args.lds_sigma)
            logging.info(f'Using LDS: [{args.lds_kernel.upper()}] ({args.lds_ks}/{args.lds_sigma})')
            if args.reweight == 'sqrt_inv':
                value_lst = np.sqrt(value_lst)
            smoothed_value = convolve1d(np.asarray(value_lst), weights=lds_kernel_window, mode='reflect')
            smoothed_value = [smoothed_value[0]] * args.bucket_start + list(smoothed_value)
            scaling = np.sum(TRAIN_BUCKET_NUM) / np.sum(np.array(TRAIN_BUCKET_NUM) / np.array(smoothed_value))
            bucket_weights = [np.float32(scaling / smoothed_value[bucket]) for bucket in range(args.bucket_num)]
        else:
            value_lst = [TRAIN_BUCKET_NUM[args.bucket_start]] * args.bucket_start + TRAIN_BUCKET_NUM[args.bucket_start:]
            if args.reweight == 'sqrt_inv':
                value_lst = np.sqrt(value_lst)
            scaling = np.sum(TRAIN_BUCKET_NUM) / np.sum(np.array(TRAIN_BUCKET_NUM) / np.array(value_lst))
            bucket_weights = [np.float32(scaling / value_lst[bucket]) for bucket in range(args.bucket_num)]

        return bucket_weights

    def get_bin_idx(self, x):
        return min(int(x * np.float32(10)), 99)

    def _get_weights(self, depth):
        sp = depth.shape
        if self.bucket_weights is not None:
            depth = depth.view(-1).cpu().numpy()
            assert depth.dtype == np.float32
            weights = np.array(list(map(lambda v: self.bucket_weights[self.get_bin_idx(v)], depth)))
            weights = torch.tensor(weights, dtype=torch.float32).view(*sp)
        else:
            weights = torch.tensor([np.float32(1.)], dtype=torch.float32).repeat(*sp)
        return weights

    def __getitem__(self, idx):
        image_name = self.frame.iloc[idx, 0]
        depth_name = self.frame.iloc[idx, 1]

        image_name = os.path.join(self.data_dir, '/'.join(image_name.split('/')[1:]))
        depth_name = os.path.join(self.data_dir, '/'.join(depth_name.split('/')[1:]))

        image = Image.open(image_name)
        depth = Image.open(depth_name)

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        sample['weight'] = self._get_weights(sample['depth'])
        sample['idx'] = idx

        if self.mask is not None:
            sample['mask'] = self.mask[idx].unsqueeze(0)

        return sample

    def __len__(self):
        return len(self.frame)


def getTrainingData(args, batch_size=64):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_training = depthDataset(data_dir=args.data_dir,
                                        csv_file=os.path.join(args.data_dir, 'nyu2_train.csv'),
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [152, 114]),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]), args=args)

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=8, pin_memory=False)

    return dataloader_training

def getTrainingFDSData(args, batch_size=64):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_training = depthDataset(data_dir=args.data_dir,
                                        csv_file=os.path.join(args.data_dir, 'nyu2_train_FDS_subset.csv'),
                                        transform=transforms.Compose([
                                            Scale(240),
                                            CenterCrop([304, 228], [152, 114]),
                                            ToTensor(),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=False, num_workers=8, pin_memory=False)
    return dataloader_training


def getTestingData(args, batch_size=64):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_testing = depthDataset(data_dir=args.data_dir,
                                       csv_file=os.path.join(args.data_dir, 'nyu2_test.csv'),
                                       mask_file=os.path.join(args.data_dir, 'test_balanced_mask.npy'),
                                       transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop([304, 228], [304, 228]),
                                           ToTensor(is_test=True),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing
