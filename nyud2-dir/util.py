import math
import logging
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

def lg10(x):
    return torch.div(torch.log(x), math.log(10))

def maxOfTwo(x, y):
    z = x.clone()
    maskYLarger = torch.lt(x, y)
    z[maskYLarger.detach()] = y[maskYLarger.detach()]
    return z

def nValid(x):
    return torch.sum(torch.eq(x, x).float())

def getNanMask(x):
    return torch.ne(x, x)

def setNanToZero(input, target):
    nanMask = getNanMask(target)
    nValidElement = nValid(target)

    _input = input.clone()
    _target = target.clone()

    _input[nanMask] = 0
    _target[nanMask] = 0

    return _input, _target, nanMask, nValidElement

class Evaluator:
    def __init__(self):
        self.shot_idx = {
            'many': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 49],
            'medium': [7, 8, 46, 48, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 63],
            'few': [0, 1, 2, 3, 4, 5, 6, 57, 59, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        }
        self.output = torch.tensor([], dtype=torch.float32)
        self.depth = torch.tensor([], dtype=torch.float32)

    def __call__(self, output, depth):
        output = output.squeeze().view(-1).cpu()
        depth = depth.squeeze().view(-1).cpu()
        self.output = torch.cat([self.output, output])
        self.depth = torch.cat([self.depth, depth])

    def evaluate_shot(self):
        metric_dict = {'overall': {}, 'many': {}, 'medium': {}, 'few': {}}
        self.depth_bucket = np.array(list(map(lambda v: self.get_bin_idx(v), self.depth.cpu().numpy())))

        for shot in metric_dict.keys():
            if shot == 'overall':
                metric_dict[shot] = self.evaluate(self.output, self.depth)
            else:
                mask = np.zeros(self.depth.size(0), dtype=np.bool)
                for i in self.shot_idx[shot]:
                    mask[np.where(self.depth_bucket == i)[0]] = True
                mask = torch.tensor(mask, dtype=torch.bool)
                metric_dict[shot] = self.evaluate(self.output[mask], self.depth[mask])

        logging.info('\n***** TEST RESULTS *****')
        for shot in ['Overall', 'Many', 'Medium', 'Few']:
            logging.info(f" * {shot}: RMSE {metric_dict[shot.lower()]['RMSE']:.3f}\t"
                        f"ABS_REL {metric_dict[shot.lower()]['ABS_REL']:.3f}\t"
                        f"LG10 {metric_dict[shot.lower()]['LG10']:.3f}\t"
                        f"MAE {metric_dict[shot.lower()]['MAE']:.3f}\t"
                        f"DELTA1 {metric_dict[shot.lower()]['DELTA1']:.3f}\t"
                        f"DELTA2 {metric_dict[shot.lower()]['DELTA2']:.3f}\t"
                        f"DELTA3 {metric_dict[shot.lower()]['DELTA3']:.3f}\t"
                        f"NUM {metric_dict[shot.lower()]['NUM']}")

        return metric_dict

    def reset(self):
        self.output = torch.tensor([], dtype=torch.float32)
        self.depth = torch.tensor([], dtype=torch.float32)

    @staticmethod
    def get_bin_idx(x):
        return min(int(x * np.float32(10)), 99)

    @staticmethod
    def evaluate(output, target):
        errors = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                  'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0, 'NUM': 0}

        _output, _target, nanMask, nValidElement = setNanToZero(output, target)

        if (nValidElement.data.cpu().numpy() > 0):
            diffMatrix = torch.abs(_output - _target)

            errors['MSE'] = torch.sum(torch.pow(diffMatrix, 2)) / nValidElement

            errors['MAE'] = torch.sum(diffMatrix) / nValidElement

            realMatrix = torch.div(diffMatrix, _target)
            realMatrix[nanMask] = 0
            errors['ABS_REL'] = torch.sum(realMatrix) / nValidElement

            LG10Matrix = torch.abs(lg10(_output) - lg10(_target))
            LG10Matrix[nanMask] = 0
            errors['LG10'] = torch.sum(LG10Matrix) / nValidElement

            yOverZ = torch.div(_output, _target)
            zOverY = torch.div(_target, _output)

            maxRatio = maxOfTwo(yOverZ, zOverY)

            errors['DELTA1'] = torch.sum(
                torch.le(maxRatio, 1.25).float()) / nValidElement
            errors['DELTA2'] = torch.sum(
                torch.le(maxRatio, math.pow(1.25, 2)).float()) / nValidElement
            errors['DELTA3'] = torch.sum(
                torch.le(maxRatio, math.pow(1.25, 3)).float()) / nValidElement

            errors['MSE'] = float(errors['MSE'].data.cpu().numpy())
            errors['ABS_REL'] = float(errors['ABS_REL'].data.cpu().numpy())
            errors['LG10'] = float(errors['LG10'].data.cpu().numpy())
            errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
            errors['DELTA1'] = float(errors['DELTA1'].data.cpu().numpy())
            errors['DELTA2'] = float(errors['DELTA2'].data.cpu().numpy())
            errors['DELTA3'] = float(errors['DELTA3'].data.cpu().numpy())
            errors['NUM'] = int(nValidElement)

        errors['RMSE'] = np.sqrt(errors['MSE'])

        return errors


def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")

def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.2, clip_max=5.):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 <= 0.).any() or (v2 < 0.).any():
        valid_pos = (((v1 > 0.) + (v2 >= 0.)) == 2)
        # print(torch.sum(valid_pos))
        factor = torch.clamp(v2[valid_pos] / v1[valid_pos], clip_min, clip_max)
        matrix[:, valid_pos] = (matrix[:, valid_pos] - m1[valid_pos]) * torch.sqrt(factor) + m2[valid_pos]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window



	
