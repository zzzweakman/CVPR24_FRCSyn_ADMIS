import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
import argparse
import struct
from torchkit.data import example_pb2


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='decode tfrecord')
    parser.add_argument('--tfrecords_dir', default=None, type=str, required=True,
                        help='path to the output of tfrecords dir path')
    parser.add_argument('--tfrecords_name', default='Syn_10k', type=str, required=True,
                        help="the index file's name")  
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='path to the output of decoded imgs')
    parser.add_argument('--limit', default=10, type=int, required=True,
                        help='limit num of decoded samples')

    args = parser.parse_args()
    return args


def parser(feature_list):
    for key, feature in feature_list:
        if key == 'image':
            image_raw = feature.bytes_list.value[0]
            return image_raw
    raise ValueError("No key=image in feature list")


def get_record(record_file, offset):
    with open(record_file, 'rb') as ifs:
        ifs.seek(offset)
        byte_len_crc = ifs.read(12)
        proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
        # proto,crc
        pb_data = ifs.read(proto_len)
        if len(pb_data) < proto_len:
            print("read pb_data err,proto_len:%s pb_data len:%s" % (proto_len, len(pb_data)))
            return None
    example = example_pb2.Example()
    example.ParseFromString(pb_data)
    # keep key value in order
    feature = sorted(example.features.feature.items())
    record = parser(feature)
    return record


def main():
    args = parse_args()
    tfrecords_dir = os.path.normpath(args.tfrecords_dir)
    tfrecords_name = args.tfrecords_name
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    limit = args.limit
    print(tfrecords_dir)
    print(tfrecords_name)

    index_file = os.path.join(tfrecords_dir, '%s.index' % tfrecords_name)
    with open(index_file, 'r') as f:
        for line_i, line in enumerate(f):
            if line_i >= limit:
                break
            raw_img_path, tfr_idx, tfr_offset = line.rstrip().split('\t')
            record_file = os.path.join(tfrecords_dir,  "%s-%05d.tfrecord" % (tfrecords_name, int(tfr_idx)))
            image_raw = get_record(record_file, int(tfr_offset))
            target_dir = os.path.join(output_dir, raw_img_path.split("/")[-2])
            if not os.path.exists( target_dir):
                os.makedirs(target_dir)
            img_path = os.path.join(target_dir, raw_img_path.split("/")[-1])
            f = open(img_path, 'wb')
            f.write(image_raw)
            f.close()

if __name__ == "__main__":
    main()
