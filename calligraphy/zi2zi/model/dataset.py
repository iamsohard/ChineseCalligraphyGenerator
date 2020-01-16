# -*- coding: utf-8 -*-

import os
import pickle as pickle
import random
import pdb
import numpy as np

from model.utils import pad_seq, bytes_to_file, \
    read_split_image, shift_and_resize_image, normalize_image


class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 1000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples


def get_batch_iter(examples, batch_size, augment, embedding_id=None):
    # the transpose ops requires deterministic
    # batch size, thus comes the padding
    # padded = pad_seq(examples, batch_size)

    def process(img):
        img = bytes_to_file(img)
        try:
            img_A, img_B = read_split_image(img)
            if augment:
                # augment the image by:
                # 1) enlarge the image
                # 2) random crop the image back to its original size
                # NOTE: image A and B needs to be in sync as how much
                # to be shifted
                w, h = img_A.shape
                multiplier = random.uniform(1.00, 1.05)
                # add an eps to prevent cropping issue
                nw = int(multiplier * w) + 1
                nh = int(multiplier * h) + 1
                shift_x = int(np.ceil(np.random.uniform(0.01, nw - w)))
                shift_y = int(np.ceil(np.random.uniform(0.01, nh - h)))
                img_A = shift_and_resize_image(img_A, shift_x, shift_y, nw, nh)
                img_B = shift_and_resize_image(img_B, shift_x, shift_y, nw, nh)
            img_A = normalize_image(img_A)
            img_B = normalize_image(img_B)
            merged = np.stack([img_A, img_B], axis=2)
            return merged
        finally:
            img.close()

    def batch_iter():
        for i in range(0, len(examples), batch_size):
            batch = examples[i: i + batch_size]
            labels = [e[0] for e in batch]
            processed = [process(e[1]) for e in batch]
            # stack into tensor
            yield labels, np.array(processed).astype(np.float32)

    def batch_iter_with_filter():
        labels, processed = [], []
        for i in range(len(examples)):
            if examples[i][0] == embedding_id:
                labels.append(embedding_id)
                processed.append(process(examples[i][1]))
            else:
                continue

            if len(labels) == batch_size:
                yield labels, np.array(processed).astype(np.float32)
                labels, processed = [], []
        if labels:
            yield labels, np.array(processed).astype(np.float32)

    if embedding_id is None:
        return batch_iter()
    else:
        return batch_iter_with_filter()


class TrainDataProvider(object):
    def __init__(self, data_dir, train_name="train.obj", val_name="val.obj", filter_by=None):
        self.data_dir = data_dir
        self.filter_by = filter_by
        self.train_path = os.path.join(self.data_dir, train_name)
        self.val_path = os.path.join(self.data_dir, val_name)
        self.train = PickledImageProvider(self.train_path)
        self.val = PickledImageProvider(self.val_path)
        if self.filter_by:
            print("filter by label ->", filter_by)
            self.train.examples = list(filter(lambda e: e[0] in self.filter_by, self.train.examples))
            self.val.examples = list(filter(lambda e: e[0] in self.filter_by, self.val.examples))
        print("train examples -> %d, val examples -> %d" % (len(self.train.examples), len(self.val.examples)))

    def get_train_iter(self, batch_size, shuffle=True):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)
        return get_batch_iter(training_examples, batch_size, augment=True)

    def get_val_iter(self, batch_size, shuffle=True):
        """
        Validation iterator runs forever
        """
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.shuffle(val_examples)
        while True:
            val_batch_iter = get_batch_iter(val_examples, batch_size, augment=False)
            for labels, examples in val_batch_iter:
                yield labels, examples

    def compute_total_batch_num(self, batch_size):
        """Total padded batch num"""
        return int(np.ceil(len(self.train.examples) / float(batch_size)))

    def get_all_labels(self):
        """Get all training labels"""
        return list({e[0] for e in self.train.examples})

    def get_train_val_path(self):
        return self.train_path, self.val_path


class InjectDataProvider(object):
    def __init__(self, obj_path):
        self.data = PickledImageProvider(obj_path)
        print("examples -> %d" % len(self.data.examples))

    def get_single_embedding_iter(self, batch_size, embedding_id):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, augment=False, embedding_id=embedding_id)
        for _, images in batch_iter:
            # inject specific embedding style here
            labels = [embedding_id] * len(images)
            yield labels, images

    def get_random_embedding_iter(self, batch_size, embedding_ids):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, augment=False)
        for _, images in batch_iter:
            # inject specific embedding style here
            labels = [random.choice(embedding_ids) for i in range(len(images))]
            yield labels, images


class NeverEndingLoopingProvider(InjectDataProvider):
    def __init__(self, obj_path):
        super(NeverEndingLoopingProvider, self).__init__(obj_path)

    def get_random_embedding_iter(self, batch_size, embedding_ids):
        while True:
            # np.random.shuffle(self.data.examples)
            rand_iter = super(NeverEndingLoopingProvider, self) \
                .get_random_embedding_iter(batch_size, embedding_ids)
            for labels, images in rand_iter:
                yield labels, images


if __name__ == '__main__':
    from PIL import Image

    pkl_images = PickledImageProvider("../binary/train.obj")
    examples = pkl_images.examples
    print(len(examples))

    b_img0 = examples[0][1]  # idx, binary
    img0 = bytes_to_file(b_img0)
    img_A, img_B = read_split_image(img0)
    img = Image.fromarray(np.uint8(img_A), "RGB")
    # img.save('my.png')

    # mat =  misc.imread(img0).astype(np.float)
    # side = int(mat.shape[1] / 2)
    # assert side * 2 == mat.shape[1]
    # img_A = mat[:, :side]  # target
    # img = Image.fromarray(np.uint8(img_B))
    img.show()
