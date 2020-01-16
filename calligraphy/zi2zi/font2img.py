# -*- coding: utf-8 -*-

import argparse
import importlib
import json
import os
import pdb
import sys

import collections
import numpy as np
from PIL import ImageFont

from model.preprocessing_helper import draw_single_char_by_font, draw_example, CHAR_SIZE, CANVAS_SIZE
from package import save_train_valid_data

importlib.reload(sys)

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None
GB775_CHARSET = None
GB6763_CHARSET = None

DEFAULT_CHARSET = "./charset/cjk_cn.json"


def load_global_charset():
    global CN_CHARSET, JP_CHARSET, KR_CHARSET, CN_T_CHARSET, GB775_CHARSET, GB6763_CHARSET
    cjk = json.load(open(DEFAULT_CHARSET))
    CN_CHARSET = cjk["gbk"]
    JP_CHARSET = cjk["jp"]
    KR_CHARSET = cjk["kr"]
    CN_T_CHARSET = cjk["gb2312_t"]
    GB775_CHARSET = cjk["gb775"]
    GB6763_CHARSET = cjk["gb6763"]


def filter_recurring_hash(charset, font, canvas_size, char_size):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = charset[:]
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char_by_font(c, font, canvas_size, char_size)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


def font2img(src, dst, charset, char_size, canvas_size,
             sample_count, sample_dir, label=0, filter_by_hash=True):
    assert os.path.isfile(src), "src file doesn't exist:%s" % src
    assert os.path.isfile(dst), "dst file doesn't exist:%s" % dst

    if not os.path.isdir(sample_dir):
        print("warning: creating sample dir: %s" % sample_dir)
        os.makedirs(sample_dir)

    src_font = ImageFont.truetype(src, size=char_size)
    dst_font = ImageFont.truetype(dst, size=char_size)

    filter_hashes = set()
    if filter_by_hash:
        filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, char_size))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0

    for c in charset:
        if count == sample_count:
            break
        e = draw_example(c, src_font, dst_font, canvas_size, filter_hashes, char_size)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)), mode='F')
            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)


load_global_charset()
parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--src_font', default='data/raw_fonts/SimSun.ttf', help='path of the source font')
parser.add_argument('--fonts_dir', default='data/raw_fonts', help='dir path of the target fonts')
parser.add_argument('--filter', type=int, default=1, help='filter recurring characters')
parser.add_argument('--charset', type=str, default='CN',
                    help='charset, can be either: CN, JP, KR , GB775, GB6763 or a one line file')
parser.add_argument('--shuffle', type=int, default=True, help='shuffle a charset before processings')
parser.add_argument('--char_size', type=int, default=CHAR_SIZE, help='character size')
parser.add_argument('--canvas_size', type=int, default=CANVAS_SIZE, help='canvas size')
parser.add_argument('--sample_count', type=int, default=1000, help='number of characters to draw')
parser.add_argument('--sample_dir', default='data/paired_images', help='directory to save examples')

# These two are for package.py
parser.add_argument('--split_ratio', type=float, default=0.05, help='split ratio between train and val')
parser.add_argument('--save_dir', default="experiments/data", help='path to save pickled files')

args = parser.parse_args()

if __name__ == "__main__":

    label = 0
    for root, dirs, files in os.walk(args.fonts_dir):
        for name in files:
            if name.lower().endswith(".ttf") and name.lower() not in ["simsun.ttf", "井柏然体.ttf"]:
                print("%s | %s" % (label, name))
                dst_font = os.path.join(root, name)

                if args.charset in ['CN', 'JP', 'KR', 'CN_T', 'GB775', 'GB6763']:
                    charset = locals().get("%s_CHARSET" % args.charset)
                else:
                    charset = [c for c in open(args.charset).readline()[:-1]]

                if args.shuffle:
                    np.random.shuffle(charset)
                font2img(args.src_font, dst_font, charset, args.char_size,
                         args.canvas_size,
                         args.sample_count, args.sample_dir, label, args.filter)
                label += 1
    print("Number of fonts:", label)

    # Save as pickled file
    save_train_valid_data(save_dir=args.save_dir, sample_dir=args.sample_dir, split_ratio=args.split_ratio)
