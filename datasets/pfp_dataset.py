from imageio import imread
from glob import glob
from tqdm import tqdm
import os.path as osp
import numpy as np
import numpy.random as npr
import re


# FIXME
DATASET_DIR = '/home/jongho/data/PennFudanPed'

# FIXME
TRAIN_RATIO = 0.4
VALID_RATIO = 0.1
# TEST_RATIO = 1 - TRAIN_RATIO - VALID_RATIO

# FIXME
CLASS_NAMES = [
    'PASpersonWalking', 'PASpersonStanding'
]

RE_LABEL = 'Original label for object (?P<idx>\d+) "(?P<cls_name>\w+)" : "PennFudanPed"'
RE_BBOX = 'Bounding box for object (?P<idx>\d+) "(?P<cls_name>\w+)" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((?P<xmin>\d+), (?P<ymin>\d+)\) - \((?P<xmax>\d+), (?P<ymax>\d+)\)'


def _extract_gt_boxes(fpath):
    with open(fpath, 'r') as f:
        all_lines = [line.strip() for line in f.readlines()]

    # Extract all objects in a sample using a regular expression.
    cls_name_map, bbox_map = dict(), dict()
    for line in all_lines:
        if re.match(RE_LABEL, line):
            gdict = re.match(RE_LABEL, line).groupdict()
            idx, cls_name = int(gdict['idx']), gdict['cls_name']
            cls_name_map[idx] = cls_name
        elif re.match(RE_BBOX, line):
            gdict = re.match(RE_BBOX, line).groupdict()
            idx, xmin, ymin, xmax, ymax = [
                int(gdict[t]) for t in ['idx', 'xmin', 'ymin', 'xmax', 'ymax']
            ]
            bbox_map[idx] = [xmin, ymin, xmax, ymax]
        else:
            continue

    gt_boxes = []
    for idx in sorted(cls_name_map.keys()):
        gt_boxes.append(bbox_map[idx] + [CLASS_NAMES.index(cls_name_map[idx])])

    return np.array(gt_boxes)


def read_data():
    all_imgs, all_img_sizes, all_gt_boxes = [], [], []

    # Get file paths of the images and annotations.
    img_dir, anno_dir = osp.join(DATASET_DIR, 'PNGImages'), osp.join(DATASET_DIR, 'Annotation')
    img_fpaths = sorted(glob(osp.join(img_dir, '*.png')))
    anno_fpaths = sorted(glob(osp.join(anno_dir, '*.txt')))

    assert len(img_fpaths) == len(anno_fpaths)

    print('Loading data instances...')
    num_samples = len(img_fpaths)
    for idx, img_fp, anno_fp in zip(tqdm(range(num_samples)), img_fpaths, anno_fpaths):
        # Samples' names without the file extension should match.
        assert osp.splitext(osp.basename(img_fp))[0] == osp.splitext(osp.basename(anno_fp))[0]

        # Load and extract the data.
        img = imread(img_fp).astype(np.float32) / 255
        img_size = img.shape[:2]
        gt_boxes = _extract_gt_boxes(anno_fp)

        all_imgs.append(img)
        all_img_sizes.append(img_size)
        all_gt_boxes.append(gt_boxes)

    data = {'image': all_imgs, 'image_size': all_img_sizes, 'gt_boxes': all_gt_boxes}

    return data


class PFPDataSet:
    def __init__(self, imgs, img_sizes, gt_boxes):
        self._obj_classes = CLASS_NAMES
        self._num_obj_classes = len(CLASS_NAMES)

        # Convert the lists to the ndarrays for later indexing by lists.
        self._imgs = np.array(imgs, dtype=np.object)
        self._img_sizes = np.array(img_sizes, dtype=np.object)
        self._gt_boxes = np.array(gt_boxes, dtype=np.object)

        self._num_samples = len(imgs)
        self._indices = np.arange(self._num_samples)

        self._reset()

    def _reset(self):
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def num_classes(self):
        return self._num_obj_classes

    @property
    def images(self):
        return self._images

    @property
    def num_samples(self):
        return self._num_samples

    def sample_batch(self, batch_size, shuffle=True):
        if shuffle:
            indices = npr.choice(self._num_samples, batch_size)
        else:
            indices = np.arange(batch_size)

        batch = {
            'image': self._imgs[indices],
            'image_size': self._img_sizes[indices],
            'gt_boxes': self._gt_boxes[indices],
        }

        return batch

    def next_batch(self, batch_size, shuffle=True):
        start_index = self._index_in_epoch

        # Shuffle the dataset, for the first epoch
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            npr.shuffle(self._indices)

        if start_index + batch_size > self._num_samples:
            # Increment the number of epochs completed.
            self._epochs_completed += 1

            # Get the rest samples in this epoch.
            num_samples_rest = self._num_samples - start_index
            indices_rest = self._indices[start_index:]

            # Shuffle the dataset, after finishing a single epoch
            if shuffle:
                npr.shuffle(self._indices)

            # Start the next epoch
            start_index = 0
            indices_new = self._indices[:batch_size - num_samples_rest]
            indices = np.concatenate([indices_rest, indices_new])
            self._index_in_epoch = batch_size - num_samples_rest
        else:
            indices = self._indices[start_index:start_index + batch_size]
            self._index_in_epoch += batch_size

        next_batch = {
            'image': self._imgs[indices],
            'image_size': self._img_sizes[indices],
            'gt_boxes': self._gt_boxes[indices],
        }

        return next_batch
