import os
from typing import Callable

import torch
from PIL import Image
from torchvision.datasets import DatasetFolder
import numpy as np

class PILImageLoader:

    def __call__(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


class SamplesWithEmbeddingsFileDataset(torch.utils.data.Dataset):

    def __init__(self,
                 samples_root: str,
                 embeddings_file_path: str,
                 sample_file_ending: str = '.png',
                 sample_loader: Callable = None,
                 sample_transform: Callable = None
                 ):
        super(SamplesWithEmbeddingsFileDataset, self).__init__()

        self.sample_loader = sample_loader
        self.transform = sample_transform

        # self.embeddings_file_path = embeddings_file_path
        self.samples = self.build_samples(
            embeddings_file_path=embeddings_file_path,
            sample_root=samples_root,
            sample_file_ending=sample_file_ending
        )

    @staticmethod
    def build_samples(embeddings_file_path: str, sample_root: str, sample_file_ending: str):
        content = torch.load(embeddings_file_path)
        samples = []
        for index, (identity, embedding_npy) in enumerate(content.items()):
            image_path = os.path.join(sample_root, identity + sample_file_ending)
            samples.append((image_path, embedding_npy))
        return samples

    def __getitem__(self, index: int):
        image_path, embedding_npy = self.samples[index]

        image = self.sample_loader(image_path)
        embedding = torch.from_numpy(embedding_npy)

        if self.transform is not None:
            image = self.transform(image)

        return image, embedding

    def __len__(self):
        return len(self.samples)


class ControlNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.data_seg = os.listdir(os.path.join(self.root_path, "CelebAMask-HQ-mask-color-256"))
        self.data_origin = os.listdir(os.path.join(self.root_path, "CelebA-HQ-128"))
        self.data_embed = np.load(os.path.join(self.root_path, "CelebA_elasticface_embeddings", "wo_crop.npy"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass
        # item = self.data[idx]

        # source_filename = item['source']
        # target_filename = item['target']
        # prompt = item['prompt']

        # source = cv2.imread('./training/fill50k/' + source_filename)
        # target = cv2.imread('./training/fill50k/' + target_filename)

        # # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0

        # # Normalize target images to [-1, 1].
        # target = (target.astype(np.float32) / 127.5) - 1.0

        # return dict(jpg=target, txt=prompt, hint=source)