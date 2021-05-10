import argparse
import time
import os
import shutil
import logging
import torch
import torch.backends.cudnn as cudnn
import loaddata
from tqdm import tqdm
from models import modules, net, resnet
from util import query_yes_no
from test import test
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='')

# training/optimization related
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--batch_size', default=32, type=int, help='batch size number') # 1 GPU - 8
parser.add_argument('--store_root', type=str, default='checkpoint')
parser.add_argument('--store_name', type=str, default='nyud2')
parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
parser.add_argument('--resume', action='store_true', default=False, help='whether to resume training')

# imbalanced related
# LDS
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=2, help='LDS gaussian/laplace kernel sigma')
# FDS
parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
parser.add_argument('--fds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
parser.add_argument('--fds_sigma', type=float, default=2, help='FDS gaussian/laplace kernel sigma')
parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
parser.add_argument('--bucket_start', type=int, default=7, help='minimum(starting) bucket for FDS, 7 for NYUDv2')
parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

# re-weighting: SQRT_INV / INV
parser.add_argument('--reweight', type=str, default='none', choices=['none', 'inverse', 'sqrt_inv'],
                    help='cost-sensitive reweighting scheme')
# two-stage training: RRT
parser.add_argument('--retrain_fc', action='store_true', default=False,
                    help='whether to retrain last regression layer (regressor)')
parser.add_argument('--pretrained', type=str, default='', help='pretrained checkpoint file path to load backbone weights for RRT')

def define_model(args):
    original_model = resnet.resnet50(pretrained=True)
    Encoder = modules.E_resnet(original_model)
    model = net.model(args, Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model

def main():
    error_best = 1e5
    metric_dict_best = {}
    epoch_best = -1

    global args
    args = parser.parse_args()

    if not args.lds and args.reweight != 'none':
        args.store_name += f'_{args.reweight}'
    if args.lds:
        args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
        if args.lds_kernel in ['gaussian', 'laplace']:
            args.store_name += f'_{args.lds_sigma}'
    if args.fds:
        args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
        if args.fds_kernel in ['gaussian', 'laplace']:
            args.store_name += f'_{args.fds_sigma}'
        args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
    if args.retrain_fc:
        args.store_name += f'_retrain_fc'
    args.store_name += f'_lr_{args.lr}_bs_{args.batch_size}'

    args.store_dir = os.path.join(args.store_root, args.store_name)

    if not args.resume:
        if os.path.exists(args.store_dir):
            if query_yes_no('overwrite previous folder: {} ?'.format(args.store_dir)):
                shutil.rmtree(args.store_dir)
                print(args.store_dir + ' removed.')
            else:
                raise RuntimeError('Output folder {} already exists'.format(args.store_dir))
        print(f"===> Creating folder: {args.store_dir}")
        os.makedirs(args.store_dir)

    logging.root.handlers = []
    log_file = os.path.join(args.store_dir, 'training_log.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ])
    logging.info(args)

    writer = SummaryWriter(args.store_dir)

    model = define_model(args)
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        model_state = torch.load(os.path.join(args.store_dir, 'checkpoint.pth.tar'))
        logging.info(f"Loading checkpoint from {os.path.join(args.store_dir, 'checkpoint.pth.tar')}"
                     f" (Epoch [{model_state['epoch']}], RMSE: {model_state['error']:.3f})")
        model.load_state_dict(model_state['state_dict'])

        args.start_epoch = model_state['epoch'] + 1
        epoch_best = model_state['epoch']
        error_best = model_state['error']
        metric_dict_best = model_state['metric']

    if args.retrain_fc:
        assert os.path.isfile(args.pretrained), f"No checkpoint found at '{args.pretrained}'"
        model_state = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state['state_dict'].items():
            if 'R' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        logging.info(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
        logging.info(f'===> Pre-trained model loaded: {args.pretrained}')
        for name, param in model.named_parameters():
            if 'R' not in name:
                param.requires_grad = False
        logging.info(f'Only optimize parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    cudnn.benchmark = True
    if not args.retrain_fc:
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)

    train_loader = loaddata.getTrainingData(args, args.batch_size)
    train_fds_loader = loaddata.getTrainingFDSData(args, args.batch_size)
    test_loader = loaddata.getTestingData(args, 1)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, train_fds_loader, model, optimizer, epoch, writer)
        error, metric_dict = test(test_loader, model)
        if error < error_best:
            error_best = error
            metric_dict_best = metric_dict
            epoch_best = epoch
            save_checkpoint(model.state_dict(), epoch, error, metric_dict, 'checkpoint_best.pth.tar')
        save_checkpoint(model.state_dict(), epoch, error, metric_dict, 'checkpoint.pth.tar')
    
    save_checkpoint(model.state_dict(), epoch, error, metric_dict, 'checkpoint_final.pth.tar')
    logging.info(f'Best epoch: {epoch_best}; RMSE: {error_best:.3f}')
    logging.info('***** TEST RESULTS *****')
    for shot in ['Overall', 'Many', 'Medium', 'Few']:
        logging.info(f" * {shot}: RMSE {metric_dict_best[shot.lower()]['RMSE']:.3f}\t"
                     f"ABS_REL {metric_dict_best[shot.lower()]['ABS_REL']:.3f}\t"
                     f"LG10 {metric_dict_best[shot.lower()]['LG10']:.3f}\t"
                     f"MAE {metric_dict_best[shot.lower()]['MAE']:.3f}\t"
                     f"DELTA1 {metric_dict_best[shot.lower()]['DELTA1']:.3f}\t"
                     f"DELTA2 {metric_dict_best[shot.lower()]['DELTA2']:.3f}\t"
                     f"DELTA3 {metric_dict_best[shot.lower()]['DELTA3']:.3f}\t"
                     f"NUM {metric_dict_best[shot.lower()]['NUM']}")

    writer.close()

def train(train_loader, train_fds_loader, model, optimizer, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, sample_batched in enumerate(train_loader):
        image, depth, weight = sample_batched['image'], sample_batched['depth'], sample_batched['weight']

        depth = depth.cuda(non_blocking=True)
        weight = weight.cuda(non_blocking=True)
        image = image.cuda()
        optimizer.zero_grad()

        if args.fds:
            output, feature = model(image, depth, epoch)
        else:
            output = model(image, depth, epoch)
        loss = torch.mean(((output - depth) ** 2) * weight)

        losses.update(loss.item(), image.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar('data/loss', loss.item(), i + epoch * len(train_loader))

        logging.info('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))

    if args.fds and epoch >= args.start_update:
        logging.info(f"Starting Creating Epoch [{epoch}] features of subsampled training data...")
        encodings, depths = [], []
        with torch.no_grad():
            for i, sample_batched in enumerate(tqdm(train_fds_loader)):
                image, depth = sample_batched['image'].cuda(), sample_batched['depth'].cuda()
                _, feature = model(image, depth, epoch)
                encodings.append(feature.data.cpu())
                depths.append(depth.data.cpu())
        encodings, depths = torch.cat(encodings, 0), torch.cat(depths, 0)
        logging.info(f"Created Epoch [{epoch}] features of subsampled training data (size: {encodings.size(0)})!")
        model.module.R.FDS.update_last_epoch_stats(epoch)
        model.module.R.FDS.update_running_stats(encodings, depths, epoch)

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state_dict, epoch, error, metric_dict, filename='checkpoint.pth.tar'):
    logging.info(f'Saving checkpoint to {os.path.join(args.store_dir, filename)}...')
    torch.save({
        'state_dict': state_dict,
        'epoch': epoch,
        'error': error,
        'metric': metric_dict
    }, os.path.join(args.store_dir, filename))

if __name__ == '__main__':
    main()
