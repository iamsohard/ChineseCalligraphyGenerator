# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import argparse
import glob
import os
import pickle as pickle
import random


def pickle_examples(paths, train_path, val_path, train_val_split=0.1):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    for path in [train_path, val_path]:
        dirpath = os.path.dirname(path)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)

    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            for p in paths:
                label = int(os.path.basename(p).split("_")[0])
                with open(p, 'rb') as f:
                    print("img %s" % p, label)
                    img_bytes = f.read()
                    r = random.random()
                    example = (label, img_bytes)
                    if r < train_val_split:
                        pickle.dump(example, fv)
                    else:
                        pickle.dump(example, ft)


def save_train_valid_data(save_dir, sample_dir, split_ratio):
    train_path = os.path.join(save_dir, "train.obj")
    val_path = os.path.join(save_dir, "val.obj")
    pickle_examples(sorted(glob.glob(os.path.join(sample_dir, "*.jpg"))), train_path=train_path, val_path=val_path,
                    train_val_split=split_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compile list of images into a pickled object for training')
    parser.add_argument('--sample_dir', required=True, help='path of examples')
    parser.add_argument('--save_dir', required=True, help='path to save pickled files')
    parser.add_argument('--split_ratio', type=float, default=0.1, help='split ratio between train and val')
    args = parser.parse_args()
    save_train_valid_data(args.save_dir, args.sample_dir, args.split_ratio)
