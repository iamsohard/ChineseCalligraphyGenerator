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
# from model.preprocessing_helper import save_imgs
from model.preprocessing_helper import draw_paired_image, CHAR_SIZE, \
    CANVAS_SIZE, draw_single_char_by_font, EMBEDDING_DIM
from model.unet import UNet
from model.utils import merge, scale_back
import scipy.misc as misc
import numpy as np
"""
People are made to have fun and be 中二 sometimes
                                --Bored Yan LeCun
"""

# parser = argparse.ArgumentParser(description='Inference for unseen data')
# parser.add_argument('--experiment_id', dest='experiment_id', type=int, default=0,
#                     help='sequence id for the experiments you prepare to run')
# parser.add_argument('--model_dir', dest='model_dir', default="experiments/checkpoint/experiment_0",
#                     help='directory that saves the model checkpoints')
# parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
# parser.add_argument('--text', type=str, default="库昊又双叒叕进三分了", help='the source images for inference')
# parser.add_argument('--embedding_id', type=int, default=67, help='embeddings involved')
# parser.add_argument('--embedding_dim', type=int, default=EMBEDDING_DIM, help="dimension for embedding")
# parser.add_argument('--save_dir', default='save_dir', type=str, help='path to save inferred images')
# parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=1,
#                     help='use conditional instance normalization in your model')
# parser.add_argument('--char_size', dest='char_size', type=int, default=CHAR_SIZE, help='character size')
# parser.add_argument('--src_font', dest='src_font', default='data/raw_fonts/SimSun.ttf', help='path of the source font')
# parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=CANVAS_SIZE, help='canvas size')
# parser.add_argument('--embedding_num', type=int, default=185,
#                     help="number for distinct embeddings")
#
# args = parser.parse_args()


def save_imgs(imgs, count, save_dir,p):
    # p = os.path.join(save_dir, "inferred_%04d.png" % count)
    save_concat_images(imgs, img_path=p)
    print("generated images saved at %s" % p)


def save_concat_images(imgs, img_path):
    cnt = 0
    for i in imgs:
        cnt+=i.shape[0]/256

    import math
    sz = math.ceil(math.sqrt(cnt))
    # print(cnt,sz*sz)

    x = list()
    for img in imgs:
        for j in range(0, int(img.shape[0]/256)):
            tmp = img[j*256:(j+1)*256,:,:]
            # print(tmp.shape)
            x.append(tmp)

    l = len(x)
    # print(l)
    # pdb.set_trace()
    concated = np.ones((sz*256, sz*256, 3))
    concated.dtype = 'float64'
    c = 0
    for i in range(0, sz):
        for j in range(0, sz):
            if c<l:

                concated[i*256:(i+1)*256, j*256:(j+1)*256, :] = x[c]
            else:
                tmp = np.ones(x[0].shape)
                tmp.dtype='float64'
                concated[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :] = tmp
            c+=1
    # print(concated.shape)
    misc.imsave(img_path, concated)

def save_imgs2(imgs, count, save_dir, p):
    # p = os.path.join(save_dir, "inferred_%04d.png" % count)
    save_concat_images2(imgs, img_path=p)
    print("generated images saved at %s" % p)

def save_concat_images2(imgs, img_path):
    concated = np.concatenate(imgs, axis=1)
    misc.imsave(img_path, concated)

    # if imgs[0].shape[0] != imgs[-1].shape[0]:
    #     diff = imgs[0].shape[0]-imgs[-1].shape[0]
    #     # print(diff)
    #     # pdb.set_trace()
    #     tmp = np.ones((diff, 256, 3))
    #     tmp.dtype = 'float64'
    #     imgs[-1] = np.concatenate((imgs[-1], tmp), axis=0)
    #     # print(imgs[-1].shape)
    # concated = np.concatenate(imgs, axis=1)
    # misc.imsave(img_path, concated)


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    src_font = ImageFont.truetype(args.src_font, size=args.char_size)

    with tf.Session(config=config) as sess:
        model = UNet(batch_size=args.batch_size, input_width=args.canvas_size, output_width=args.canvas_size,
                     experiment_id=args.experiment_id, embedding_dim=args.embedding_dim,
                     embedding_num=args.embedding_num)
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
            print("getshape",type(merged_fake_images),merged_fake_images.shape)
            if len(batch_buffer)>0 and merged_fake_images.shape!=batch_buffer[0].shape:

                continue
            batch_buffer.append(merged_fake_images)
            # if len(batch_buffer) == 10:
            #     save_imgs(batch_buffer, count, args.save_dir)
            #     batch_buffer = list()
            count += 1


        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count, args.save_dir)





def infer_by_text_api(str, embedding_id,path):
    # CUDA_VISIBLE_DEVICES=0
    # --model_dir=experiments/checkpoint/experiment_0
    # --batch_size=32
    # --embedding_id=67
    # --save_dir=save_dir

    rootpath = os.path.dirname(os.path.abspath(__file__))
    print(rootpath)
    # path = os.path.join(rootpath, 'zi2zi')

    # default
    experiment_id = 0
    model_dir = os.path.join(rootpath, "experiments/checkpoint/experiment_0")
    batch_size = 16
    text = "库昊又双叒叕进三分了"
    # embedding_id = 67
    embedding_dim = EMBEDDING_DIM
    # save_dir = os.path.join(rootpath, 'save_dir')
    inst_norm = 1
    char_size = CHAR_SIZE
    src_font = os.path.join(rootpath, 'data/raw_fonts/SimSun.ttf')
    canvas_size = CANVAS_SIZE
    embedding_num = 185

    # ours
    text = str
    batch_size = 32
    model_dir = os.path.join(rootpath, "experiments/checkpoint/experiment_0")
    # embedding_id = 67
    save_dir = os.path.join(rootpath, "save_dir")

    # print(str, path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    src_font = ImageFont.truetype(src_font, size=char_size)

    with tf.Session(config=config) as sess:
        model = UNet(batch_size=batch_size, input_width=canvas_size, output_width=canvas_size,
                     experiment_id=experiment_id, embedding_dim=embedding_dim,
                     embedding_num=embedding_num)
        model.register_session(sess)
        model.build_model(is_training=False, inst_norm=inst_norm)
        model.load_model(model_dir)

        count = 0
        batch_buffer = list()
        examples = []
        for ch in list(text):
            src_img = draw_single_char_by_font(ch, src_font, canvas_size, char_size)

            paired_img = draw_paired_image(src_img, src_img, canvas_size)

            # p = os.path.join(save_dir, "inferred_%04d.png" % 100)
            # p = path
            # misc.imsave(p, paired_img)

            buffered = BytesIO()
            paired_img.save(buffered, format="JPEG")

            examples.append((embedding_id, buffered.getvalue()))
        batch_iter = get_batch_iter(examples, batch_size, augment=False)

        for _, images in batch_iter:
            # inject specific embedding style here
            labels = [embedding_id] * len(images)

            fake_imgs = model.generate_fake_samples(images, labels)[0]
            merged_fake_images = merge(scale_back(fake_imgs), [-1, 1])  # scale 0-1
            # print("getshape", type(merged_fake_images), merged_fake_images.shape)
            # if len(batch_buffer) > 0 and merged_fake_images.shape != batch_buffer[0].shape:
            #     continue
            batch_buffer.append(merged_fake_images)
            # print("getshape",merged_fake_images.shape)
            # if len(batch_buffer) == 10:
            #     save_imgs(batch_buffer, count, save_dir, path)
            #     batch_buffer = list()
            # count += 1

        if batch_buffer:
            # last batch
            # l = len(batch_buffer)
            # for i in range(l, 10):
            #     batch_buffer.append(np.ones(81))
            save_imgs(batch_buffer, count, save_dir, path)

    return path


def infer_by_text_api2(str, str2, embedding_id,path):
    # CUDA_VISIBLE_DEVICES=0
    # --model_dir=experiments/checkpoint/experiment_0
    # --batch_size=32
    # --embedding_id=67
    # --save_dir=save_dir

    rootpath = os.path.dirname(os.path.abspath(__file__))
    print(rootpath)
    # path = os.path.join(rootpath, 'zi2zi')

    # default
    experiment_id = 0
    model_dir = os.path.join(rootpath, "experiments/checkpoint/experiment_0")
    batch_size = 16
    text = "库昊又双叒叕进三分了"
    # embedding_id = 67
    embedding_dim = EMBEDDING_DIM
    # save_dir = os.path.join(rootpath, 'save_dir')
    inst_norm = 1
    char_size = CHAR_SIZE
    src_font = os.path.join(rootpath, 'data/raw_fonts/SimSun.ttf')
    canvas_size = CANVAS_SIZE
    embedding_num = 185

    # ours
    text = str

    batch_size = 32
    model_dir = os.path.join(rootpath, "experiments/checkpoint/experiment_0")
    # embedding_id = 67
    save_dir = os.path.join(rootpath, "save_dir")

    # print(str, path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    src_font = ImageFont.truetype(src_font, size=char_size)

    with tf.Session(config=config) as sess:
        model = UNet(batch_size=batch_size, input_width=canvas_size, output_width=canvas_size,
                     experiment_id=experiment_id, embedding_dim=embedding_dim,
                     embedding_num=embedding_num)
        model.register_session(sess)
        model.build_model(is_training=False, inst_norm=inst_norm)
        model.load_model(model_dir)


        count = 0
        batch_buffer = list()
        examples = []

        for ch in list(text):
            src_img = draw_single_char_by_font(ch, src_font, canvas_size, char_size)

            paired_img = draw_paired_image(src_img, src_img, canvas_size)

            # p = os.path.join(save_dir, "inferred_%04d.png" % 100)
            # p = path
            # misc.imsave(p, paired_img)

            buffered = BytesIO()
            paired_img.save(buffered, format="JPEG")

            examples.append((embedding_id, buffered.getvalue()))
        batch_iter1 = get_batch_iter(examples, batch_size, augment=False)

        examples = []
        for ch in list(str2):
            src_img = draw_single_char_by_font(ch, src_font, canvas_size, char_size)

            paired_img = draw_paired_image(src_img, src_img, canvas_size)

            # p = os.path.join(save_dir, "inferred_%04d.png" % 100)
            # p = path
            # misc.imsave(p, paired_img)

            buffered = BytesIO()
            paired_img.save(buffered, format="JPEG")

            examples.append((embedding_id, buffered.getvalue()))
        batch_iter2 = get_batch_iter(examples, batch_size, augment=False)


        for _, images in batch_iter1:
            # inject specific embedding style here
            labels = [embedding_id] * len(images)

            fake_imgs = model.generate_fake_samples(images, labels)[0]
            merged_fake_images = merge(scale_back(fake_imgs), [-1, 1])  # scale 0-1
            # print("getshape", type(merged_fake_images), merged_fake_images.shape)
            # if len(batch_buffer) > 0 and merged_fake_images.shape != batch_buffer[0].shape:
            #     continue
            batch_buffer.append(merged_fake_images)
            # print("getshape",merged_fake_images.shape)
            # if len(batch_buffer) == 10:
            #     save_imgs(batch_buffer, count, save_dir, path)
            #     batch_buffer = list()
            # count += 1
        for _, images in batch_iter2:
            # inject specific embedding style here
            labels = [embedding_id] * len(images)

            fake_imgs = model.generate_fake_samples(images, labels)[0]
            merged_fake_images = merge(scale_back(fake_imgs), [-1, 1])  # scale 0-1
            # print("getshape", type(merged_fake_images), merged_fake_images.shape)
            # if len(batch_buffer) > 0 and merged_fake_images.shape != batch_buffer[0].shape:
            #     continue
            batch_buffer.append(merged_fake_images)
            # print("getshape",merged_fake_images.shape)
            # if len(batch_buffer) == 10:
            #     save_imgs(batch_buffer, count, save_dir, path)
            #     batch_buffer = list()
            # count += 1

        if batch_buffer:
            # last batch
            # l = len(batch_buffer)
            # for i in range(l, 10):
            #     batch_buffer.append(np.ones(81))
            save_imgs2(batch_buffer, count, save_dir, path)

        model = None


    return path

if __name__ == '__main__':
    # tf.app.run()
    str="永和九年，岁在癸丑"
    path = '/media/Data/fanglingfei/MyDjango/Calligraphy/calligraphy/static/calligraphy/genimgs/20200115212916.jpg'
    infer_by_text_api(str,9,path)
