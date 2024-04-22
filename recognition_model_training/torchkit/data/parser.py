import cv2
import numpy as np
import dareblopy as db
from . import example_pb2
from torchvision.transforms import functional as F
from PIL import Image

class IndexParser(object):
    """ Class for Line parser.
    """

    def __init__(self) -> None:
        self.sample_num = 0
        self.class_num = 0

    def __call__(self, line):
        line_s = line.rstrip().split('\t')
        if len(line_s) == 2:
            # Default line format
            img_path, label = line_s
            label = int(label)
            self.sample_num += 1
            self.class_num = max(self.class_num, label)
            return (img_path, label)
        elif len(line_s) == 4:
            # IndexTFRDataset line format
            tfr_name, tfr_index, tfr_offset, label = line_s
            label = int(label)
            tfr_file = "{0}/{0}-{1:05d}.tfrecord".format(tfr_name, int(tfr_index))
            tfr_offset = int(tfr_offset)
            self.sample_num += 1
            self.class_num = max(self.class_num, label)
            return (tfr_file, tfr_offset, label)
        else:
            raise RuntimeError("IndexParser line length %d not supported" % len(line_s))

    def reset(self):
        self.sample_num = 0
        self.class_num = 0


class ImgSampleParser(object):
    """ Class for Image Sample parser
    """

    def __init__(self, transform, crop_augmentation_prob=0.2) -> None:
        self.transform = transform
        self.crop_augmentation_prob = crop_augmentation_prob
    def __call__(self, path, label):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            # before augment: flip
            image = self.transform[0](image)
            # adaface augment: crop
            if np.random.random() < self.crop_augmentation_prob:
                ### method:1 ###
                image = self.transform[1](image)
                ### method:2 ###
                # image = self.crop_augment(image)
            # after augment: normalization
            image = self.transform[2](image)
        return image, label

# AdaFace augment
class AdaFaceTFRecordSampleParser(object):
    """ Class for TFRecord Sample parser
    """

    def __init__(self, transform, crop_augmentation_prob = 0.2) -> None:
        self.transform = transform
        self.crop_augmentation_prob = crop_augmentation_prob
        self.file_readers = dict()
    
    def crop_augment(self, sample):
        new = np.zeros_like(np.array(sample))
        i, j, h, w = self.transform[1].get_params(sample,self.transform[1].scale,self.transform[1].ratio)
        cropped = F.crop(sample, i, j, h, w)
        new[i:i+h,j:j+w, :] = np.array(cropped)
        sample = Image.fromarray(new.astype(np.uint8))
        return sample

    def __call__(self, record_path, offset, label):
        rr = self.file_readers.get(record_path, None)
        if rr is None:
            rr = db.RecordReader(record_path)
            self.file_readers[record_path] = rr
        pb_data = rr.read_record(offset)
        example = example_pb2.Example()
        example.ParseFromString(pb_data)
        image_raw = example.features.feature['image'].bytes_list.value[0]
        image = cv2.imdecode(np.frombuffer(image_raw, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            # before augment: flip
            image = self.transform[0](image)
            # adaface augment: crop
            if np.random.random() < self.crop_augmentation_prob:
                ### method:1 ###
                image = self.transform[1](image)
                ### method:2 ###
                # image = self.crop_augment(image)
            # after augment: normalization
            image = self.transform[2](image)
        return image, label



class TFRecordSampleParser(object):
    """ Class for TFRecord Sample parser
    """

    def __init__(self, transform) -> None:
        self.transform = transform
        self.file_readers = dict()

    def __call__(self, record_path, offset, label):
        rr = self.file_readers.get(record_path, None)
        if rr is None:
            rr = db.RecordReader(record_path)
            self.file_readers[record_path] = rr
        pb_data = rr.read_record(offset)
        example = example_pb2.Example()
        example.ParseFromString(pb_data)
        image_raw = example.features.feature['image'].bytes_list.value[0]
        image = cv2.imdecode(np.frombuffer(image_raw, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class Synthetic_IndexParser(object):

    """ Class for Line parser.
    """
    def __init__(self):
        self.sample_num = 0
        self.class_num = 0
    # don't include root path 
    def __call__(self, label, name, file_index, file_offset):
        # IndexTFRDataset line format
        label = int(label)
        tfr_file = "{0}-{1:05d}.tfrecord".format(name, int(file_index))
        tfr_offset = int(file_offset)
        self.sample_num += 1
        self.class_num = max(self.class_num, label)
        return (tfr_file, tfr_offset, label)
   
    def reset(self):
        self.sample_num = 0
        self.class_num = 0
