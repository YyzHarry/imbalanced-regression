import os
import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel

import loaddata
from models import modules, net, resnet
from util import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_model', type=str, default='', help='evaluation model path')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    args = parser.parse_args()

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler()
        ])

    model = define_model()
    assert os.path.isfile(args.eval_model), f"No checkpoint found at '{args.eval_model}'"
    model = torch.nn.DataParallel(model).cuda()
    model_state = torch.load(args.eval_model)
    logging.info(f"Loading checkpoint from {args.eval_model}")
    model.load_state_dict(model_state['state_dict'], strict=False)
    logging.info('Loaded successfully!')

    test_loader = loaddata.getTestingData(args, 1)
    test(test_loader, model)

def test(test_loader, model):
    model.eval()

    logging.info('Starting testing...')

    evaluator = Evaluator()

    with torch.no_grad():
        for i, sample_batched in enumerate(tqdm(test_loader)):
            image, depth, mask = sample_batched['image'], sample_batched['depth'], sample_batched['mask']
            depth = depth.cuda(non_blocking=True)
            image = image.cuda()
            output = model(image)
            output = nn.functional.interpolate(output, size=[depth.size(2),depth.size(3)], mode='bilinear', align_corners=True)

            evaluator(output[mask], depth[mask])

    logging.info('Finished testing. Start printing statistics below...')
    metric_dict = evaluator.evaluate_shot()

    return metric_dict['overall']['RMSE'], metric_dict

def define_model():
    original_model = resnet.resnet50(pretrained = True)
    Encoder = modules.E_resnet(original_model)
    model = net.model(None, Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model

if __name__ == '__main__':
    main()
