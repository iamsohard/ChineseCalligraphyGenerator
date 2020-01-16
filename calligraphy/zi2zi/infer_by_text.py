# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
from io import BytesIO
import pdb

import tensorflow as tf
from PIL import ImageFont

from model.dataset import get_batch_iter
from model.preprocessing_helper import save_imgs, draw_paired_image, CHAR_SIZE, \
    CANVAS_SIZE, draw_single_char_by_font, EMBEDDING_DIM
from model.unet import UNet
from model.utils import merge, scale_back
import scipy.misc as misc
"""
People are made to have fun and be 中二 sometimes
                                --Bored Yan LeCun
"""

parser = argparse.ArgumentParser(description='Inference for unseen data')
parser.add_argument('--experiment_id', dest='experiment_id', type=int, default=0,
                    help='sequence id for the experiments you prepare to run')
parser.add_argument('--model_dir', dest='model_dir', default="experiments/checkpoint/experiment_0",
                    help='directory that saves the model checkpoints')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--text', type=str, default="库昊又双叒叕进三分了", help='the source images for inference')
parser.add_argument('--embedding_id', type=int, default=67, help='embeddings involved')
parser.add_argument('--embedding_dim', type=int, default=EMBEDDING_DIM, help="dimension for embedding")
parser.add_argument('--save_dir', default='save_dir', type=str, help='path to save inferred images')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=1,
                    help='use conditional instance normalization in your model')
parser.add_argument('--char_size', dest='char_size', type=int, default=CHAR_SIZE, help='character size')
parser.add_argument('--src_font', dest='src_font', default='data/raw_fonts/SimSun.ttf', help='path of the source font')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=CANVAS_SIZE, help='canvas size')
parser.add_argument('--embedding_num', type=int, default=185,
                            help="number for distinct embeddings")

args = parser.parse_args()


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    src_font = ImageFont.truetype(args.src_font, size=args.char_size)

    with tf.Session(config=config) as sess:
        model = UNet(batch_size=args.batch_size, input_width=args.canvas_size, output_width=args.canvas_size,
                     experiment_id=args.experiment_id, embedding_dim=args.embedding_dim,embedding_num=args.embedding_num )
        model.register_session(sess)
        model.build_model(is_training=False, inst_norm=args.inst_norm)
        model.load_model(args.model_dir)

        count = 0
        batch_buffer = list()
        examples = []
        for ch in list(args.text):
            src_img = draw_single_char_by_font(ch, src_font, args.canvas_size, args.char_size)
            
            paired_img = draw_paired_image(src_img, src_img, args.canvas_size)
            
            p = os.path.join(args.save_dir, "inferred_%04d.png" % 100)
            misc.imsave(p, paired_img)
            
            buffered = BytesIO()
            paired_img.save(buffered, format="JPEG")

            examples.append((args.embedding_id, buffered.getvalue()))
        batch_iter = get_batch_iter(examples, args.batch_size, augment=False)

        for _, images in batch_iter:
            # inject specific embedding style here
            labels = [args.embedding_id] * len(images)

            fake_imgs = model.generate_fake_samples(images, labels)[0]
            merged_fake_images = merge(scale_back(fake_imgs), [-1, 1])  # scale 0-1
            batch_buffer.append(merged_fake_images)
            if len(batch_buffer) == 10:
                save_imgs(batch_buffer, count, args.save_dir)
                batch_buffer = list()
            count += 1

        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count, args.save_dir)


if __name__ == '__main__':
    tf.app.run()
