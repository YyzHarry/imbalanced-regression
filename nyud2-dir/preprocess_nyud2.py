import os
import argparse
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from nyu_transform import *
from loaddata import depthDataset

def load_data(args):
    train_dataset = depthDataset(
        csv_file=os.path.join(args.data_dir, 'nyu2_train.csv'),
        transform=transforms.Compose([
            Scale(240),
            CenterCrop([304, 228], [304, 228]),
            ToTensor(is_test=False),
        ])
    )
    train_dataloader = DataLoader(train_dataset, 256, shuffle=False, num_workers=16, pin_memory=False)

    test_dataset = depthDataset(
        csv_file=os.path.join(args.data_dir, 'nyu2_test.csv'),
        transform=transforms.Compose([
            Scale(240),
            CenterCrop([304, 228], [304, 228]),
            ToTensor(is_test=True),
        ])
    )
    # print(train_dataset.__len__(), test_dataset.__len__())
    test_dataloader = DataLoader(test_dataset, 256, shuffle=False, num_workers=16, pin_memory=False)

    return train_dataloader, test_dataloader

def create_FDS_train_subset_id(args):
    print('Creating FDS statistics updating subset ids...')
    train_dataloader, _ = load_data(args)
    train_depth_values = []
    for i, sample in enumerate(tqdm(train_dataloader)):
        train_depth_values.append(sample['depth'].squeeze())
    train_depth_values = torch.cat(train_depth_values, 0)
    select_idx = np.random.choice(a=list(range(train_depth_values.size(0))), size=600, replace=False)
    np.save(os.path.join(args.data_dir, 'FDS_train_subset_id.npy'), select_idx)

def create_FDS_train_subset(args):
    print('Creating FDS statistics updating subset...')
    frame = pd.read_csv(os.path.join(args.data_dir, 'nyu2_train.csv'), header=None)
    select_id = np.load(os.path.join(args.data_dir, 'FDS_train_subset_id.npy'))
    frame.iloc[select_id].to_csv(os.path.join(args.data_dir, 'nyu2_train_FDS_subset.csv'), index=False, header=False)

def get_bin_idx(x):
    return min(int(x * np.float32(10)), 99)

def create_balanced_testset(args):
    print('Creating balanced test set mask...')
    _, test_dataloader = load_data(args)
    test_depth_values = []

    for i, sample in enumerate(tqdm(test_dataloader)):
        test_depth_values.append(sample['depth'].squeeze())
    test_depth_values = torch.cat(test_depth_values, 0)
    test_depth_values_flatten = test_depth_values.view(-1).numpy()
    test_bins_number, _ = np.histogram(a=test_depth_values_flatten, bins=100, range=(0., 10.))

    select_pixel_num = min(test_bins_number[test_bins_number != 0])
    test_depth_values_flatten_bins = np.array(list(map(lambda v: get_bin_idx(v), test_depth_values_flatten)))

    test_depth_flatten_mask = np.zeros(test_depth_values_flatten.shape[0], dtype=np.uint8)
    for i in range(7, 100):
        bucket_idx = np.where(test_depth_values_flatten_bins == i)[0]
        select_bucket_idx = np.random.choice(a=bucket_idx, size=select_pixel_num, replace=False)
        test_depth_flatten_mask[select_bucket_idx] = np.uint8(1)
    test_depth_mask = test_depth_flatten_mask.reshape(test_depth_values.numpy().shape)
    np.save(os.path.join(args.data_dir, 'test_balanced_mask.npy'), test_depth_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    args = parser.parse_args()

    create_FDS_train_subset_id(args)
    create_FDS_train_subset(args)
    create_balanced_testset(args)