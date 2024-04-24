import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from skimage import transform as trans
from datetime import datetime as dt


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='imgs to tfrecord')
    parser.add_argument('--img_list', default="../../dataset/Syn_data/images", type=str, 
                        help='path to the image file')
    parser.add_argument('--tfrecords_dir', default='../../dataset/Syn_data/TFR_training_data/', type=str,  required=False,
                        help='path to the output of tfrecords dir path')
    parser.add_argument('--tfrecords_name', default='Syn_10k', type=str,  required=False,
                        help='file name')
    args = parser.parse_args()
    return args




def get_all_files(root, extension_list=['.png', '.jpg', '.jpeg'], sort=False):

    # all_files = list()
    # for (dirpath, dirnames, filenames) in os.walk(root):
    #     all_files += [os.path.join(dirpath, file) for file in filenames]
    
    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(root):
        # 增加文件夹计数器
        temp = sorted([os.path.join(dirpath, file) for file in filenames])
        if len(temp) != 0:
            # for competition: 49 + 5 = 54 images per id
            all_files += temp[1:]

    if extension_list is None:
        return all_files
    all_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, all_files))
    if sort:
        all_files = natural_sort(all_files)   
    print(len(all_files))
    return all_files


def main():
    args = parse_args()
    tfrecords_dir = args.tfrecords_dir
    tfrecords_name = args.tfrecords_name
    if not os.path.isdir(tfrecords_dir):
        os.makedirs(tfrecords_dir)
    count = 0
    cur_shard_size = 0
    cur_shard_idx = -1
    cur_shard_writer = None
    cur_shard_path = None
    cur_shard_offset = None
    idx_writer = open(os.path.join(tfrecords_dir, "%s.index" % tfrecords_name), 'w')
    f = get_all_files(args.img_list)
    # with open(args.img_list, 'r') as f:
    for line in f:
        img_path = line.rstrip()
        img = cv2.imread(img_path)
        img_bytes = cv2.imencode('.jpg', img)[1].tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}))

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
        idx_writer.write('{}\t{}\t{}\n'.format(img_path, cur_shard_idx, cur_shard_offset))
        cur_shard_offset += (len(example_bytes) + 16)
        if cur_shard_size % 5000 == 0:
            print("%d / %d"%(cur_shard_size, 500000))

        count += 1
        cur_shard_size = (cur_shard_size + 1) % 500000
    if cur_shard_writer is not None:
        cur_shard_writer.close()
    idx_writer.close()
    print('total examples number = {}'.format(count))
    print('total shard number = {}'.format(cur_shard_idx+1))


if __name__ == '__main__':
    main()
