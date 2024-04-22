import os
import sys
import cv2
import argparse
import io
import numpy as np
import tensorflow as tf
import mxnet as mx
import PIL.Image
from datetime import datetime as dt


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='data path information')
    parser.add_argument('--bin_path', default='faces_webface_112x112/train.rec', type=str,
                        help='path to the binary image file')
    parser.add_argument('--idx_path', default='faces_webface_112x112/train.idx', type=str,
                        help='path to the image index path')
    parser.add_argument('--tfrecords_name', default='TFR-CASIA_webface', type=str,
                        help='path to the output of tfrecords dir path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_shape = (3, 112, 112)
    print(tf.__version__)
    imgrec = mx.recordio.MXIndexedRecordIO(args.idx_path, args.bin_path, 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    print(header.label)
    imgidx = list(range(1, int(header.label[0])))

    tfrecords_dir = os.path.join('./', args.tfrecords_name)
    tfrecords_name = args.tfrecords_name
    if not os.path.isdir(tfrecords_dir):
        os.makedirs(tfrecords_dir)

    idx_file = os.path.join(tfrecords_dir, '{}.index'.format(tfrecords_name))
    idx_writer = open(idx_file, 'w')

    count = 0
    cur_shard_size = 0
    cur_shard_idx = -1
    cur_shard_writer = None
    cur_shard_path = None
    cur_shard_offset = None
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        label_int = int(header.label)  # int(header.label[0]) for BUPT
        label = np.array(int(label_int), dtype=np.int32).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label, ])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img, ]))}))

        if cur_shard_size == 0:
            print("{}: {} processed".format(dt.now(), count))
            cur_shard_idx += 1
            record_filename = '{0}-{1:05}.tfrecord'.format(tfrecords_name, cur_shard_idx)
            if cur_shard_writer is not None:
                cur_shard_writer.close()
            cur_shard_path = os.path.join(tfrecords_dir, record_filename)
            cur_shard_writer = tf.compat.v1.python_io.TFRecordWriter(cur_shard_path)
            cur_shard_offset = 0

        example_bytes = example.SerializeToString()
        cur_shard_writer.write(example_bytes)
        cur_shard_writer.flush()
        idx_writer.write('{}\t{}\t{}\t{}\n'.format(tfrecords_name, cur_shard_idx, cur_shard_offset, label_int))
        # idx_writer.write('{}\t{}\t{}\n'.format(tfrecords_name, cur_shard_idx, cur_shard_offset))
        cur_shard_offset += (len(example_bytes) + 16)

        count += 1
        cur_shard_size = (cur_shard_size + 1) % 500000

    if cur_shard_writer is not None:
        cur_shard_writer.close()
    print('total examples number = {}'.format(count))
    print('total shard number = {}'.format(cur_shard_idx + 1))


if __name__ == '__main__':
    main()
