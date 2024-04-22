import os
from typing import Callable

import torch
from PIL import Image
from torchvision.datasets import DatasetFolder


class PILImageLoader:

    def __call__(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


class SamplesWithEmbeddingsDDIM(torch.utils.data.Dataset):

    def __init__(self,
                 samples_root: str,
                 embeddings_file_path: str,
                 sample_file_ending: str = '.png',
                 sample_loader: Callable = None,
                 sample_transform: Callable = None
                 ):
        super(SamplesWithEmbeddingsDDIM, self).__init__()

        self.sample_loader = sample_loader
        self.transform = sample_transform
        self.embeddings_file_path = embeddings_file_path
        self.samples_root = samples_root
        self.samples = self.build_samples(
            embeddings_file_path=embeddings_file_path,
            sample_root=samples_root,
            sample_file_ending=sample_file_ending
        )

    @staticmethod
    def build_samples(embeddings_file_path: str, sample_root: str, sample_file_ending: str):
        images = os.listdir(sample_root)
        # 序号升序
        images.sort()
        samples = []
        for name in images:
            # if index in [28, 35, 38, 60, 63, 120, 121, 407, 847, 910]:
            image_path = os.path.join(sample_root, name)
            samples.append(image_path)
        print("the number of latent images: %d"%len(samples))
        return samples

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        image = self.sample_loader(image_path)
        if self.transform is not None:
            image = self.transform(image)

        return image
    def __len__(self):
        return len(self.samples)
