from train_val import train_net
from config import cfg, cfg_from_file, cfg_from_list
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
from datetime import datetime
import time
import argparse
import numpy as np
import os
import os.path as osp


ROOT_DIR = osp.abspath(osp.dirname(__file__))

PFP_TRAIN_RATIO = 0.4
PFP_VALID_RATIO = 0.1


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--dataset',
                      help='dataset name',
                      default='pfp', type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  args = parser.parse_args()
  return args


def build_datasets(name, train_ratio, valid_ratio):
    if name == 'pfp':
        from datasets.pfp_dataset import read_data, PFPDataSet as Dataset
    else:
        raise ValueError('Unexpected dataset: {}'.format(name))

    data = read_data()
    num_train = int(len(data['image']) * train_ratio)
    num_valid = int(len(data['image']) * valid_ratio)
    train_dataset = Dataset(data['image'][:num_train], data['image_size'][:num_train],
                            data['gt_boxes'][:num_train])
    valid_dataset = Dataset(data['image'][num_train:num_train + num_valid],
                            data['image_size'][num_train:num_train + num_valid],
                            data['gt_boxes'][num_train:num_train + num_valid])

    return train_dataset, valid_dataset


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
      cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
      cfg_from_list(args.set_cfgs)

    np.random.seed(cfg.RNG_SEED)

    # Prepare the datasets for training and validation.
    train_dataset, valid_dataset = build_datasets('pfp', PFP_TRAIN_RATIO, PFP_VALID_RATIO)

    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = osp.join(ROOT_DIR, 'exp/{}'.format(timestamp))
    if not osp.isdir(output_dir):
        os.makedirs(output_dir)

    # Load network.
    if args.net == 'vgg16':
        net = vgg16()
    elif args.net == 'res50':
        net = resnetv1(num_layers=50)
    elif args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'res152':
        net = resnetv1(num_layers=152)
    elif args.net == 'mobile':
        net = mobilenetv1()
    else:
        raise NotImplementedError

    train_net(
        net, train_dataset, valid_dataset, output_dir, pretrained_model=args.weight,
        max_iters=args.max_iters
    )
