import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from scipy.ndimage import gaussian_filter1d
from util import calibrate_mean_var
from scipy.signal.windows import triang

class FDS(nn.Module):

    def __init__(self, feature_dim, bucket_num=50, bucket_start=0, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=2, momentum=0.9):
        super(FDS, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.kernel_window = self._get_kernel_window(kernel, ks, sigma)
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        self.register_buffer('epoch', torch.zeros(1).fill_(start_update))
        self.register_buffer('running_mean', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start))

    @staticmethod
    def _get_kernel_window(kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(
                map(laplace, np.arange(-half_ks, half_ks + 1)))

        logging.info(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')
        return torch.tensor(kernel_window, dtype=torch.float32).cuda()

    def _get_bucket_idx(self, label):
        label = np.float32(label)
        _, bins_edges = np.histogram(a=np.array([], dtype=np.float32), bins=self.bucket_num, range=(0., 5.))
        if label == 5.:
            return self.bucket_num - 1
        else:
            return max(np.where(bins_edges > label)[0][0] - 1, self.bucket_start)

    def _update_last_epoch_stats(self):
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        self.smoothed_mean_last_epoch = F.conv1d(
            input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        self.smoothed_var_last_epoch = F.conv1d(
            input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_last_epoch_stats()
            logging.info(f"Updated smoothed statistics of last epoch on Epoch [{epoch}]!")

    def update_running_stats(self, features, labels, epoch):
        if epoch < self.epoch:
            return

        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        buckets = np.array([self._get_bucket_idx(label) for label in labels])
        for bucket in np.unique(buckets):
            curr_feats = features[torch.tensor((buckets == bucket).astype(np.uint8))]
            curr_num_sample = curr_feats.size(0)
            curr_mean = torch.mean(curr_feats, 0)
            curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

            self.num_samples_tracked[bucket - self.bucket_start] += curr_num_sample
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[bucket - self.bucket_start]))
            factor = 0 if epoch == self.start_update else factor
            self.running_mean[bucket - self.bucket_start] = \
                (1 - factor) * curr_mean + factor * self.running_mean[bucket - self.bucket_start]
            self.running_var[bucket - self.bucket_start] = \
                (1 - factor) * curr_var + factor * self.running_var[bucket - self.bucket_start]

        # make up for zero training samples buckets
        for bucket in range(self.bucket_start, self.bucket_num):
            if bucket not in np.unique(buckets):
                if bucket == self.bucket_start:
                    self.running_mean[0] = self.running_mean[1]
                    self.running_var[0] = self.running_var[1]
                elif bucket == self.bucket_num - 1:
                    self.running_mean[bucket - self.bucket_start] = self.running_mean[bucket - self.bucket_start - 1]
                    self.running_var[bucket - self.bucket_start] = self.running_var[bucket - self.bucket_start - 1]
                else:
                    self.running_mean[bucket - self.bucket_start] = (self.running_mean[bucket - self.bucket_start - 1] +
                                                                     self.running_mean[bucket - self.bucket_start + 1]) / 2.
                    self.running_var[bucket - self.bucket_start] = (self.running_var[bucket - self.bucket_start - 1] +
                                                                    self.running_var[bucket - self.bucket_start + 1]) / 2.
        logging.info(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features, labels, epoch):
        if epoch < self.start_smooth:
            return features

        labels = labels.squeeze(1)
        buckets = np.array([self._get_bucket_idx(label) for label in labels])
        for bucket in np.unique(buckets):
            features[torch.tensor((buckets == bucket).astype(np.uint8))] = calibrate_mean_var(
                features[torch.tensor((buckets == bucket).astype(np.uint8))],
                self.running_mean_last_epoch[bucket - self.bucket_start],
                self.running_var_last_epoch[bucket - self.bucket_start],
                self.smoothed_mean_last_epoch[bucket - self.bucket_start],
                self.smoothed_var_last_epoch[bucket - self.bucket_start]
            )

        return features
