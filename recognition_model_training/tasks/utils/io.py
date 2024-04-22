import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchkit.data.sampler import DummySampler
from torchkit.data import MultiDataset


def make_inputs(data_path='/remote-home/share/yxmi/datasets/TFR-BUPT/',
                idx_path='/remote-home/share/yxmi/datasets/TFR-BUPT/',
                batch_size=32):
    rgb_mean = [0.5, 0.5, 0.5]
    rgb_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std)
    ])

    ds_names = ['TFR-BUPT']
    ds = MultiDataset(data_path, idx_path,
                      ds_names, transform)
    ds.make_dataset(shard=False)

    sampler = DummySampler(ds, [batch_size])
    train_loader = DataLoader(ds, sum([batch_size]), shuffle=False,
                              num_workers=1, pin_memory=True,
                              sampler=sampler, drop_last=False)

    return train_loader


def draw_image_batch(dtype='tensor'):
    dataloader = make_inputs()

    for step, samples in enumerate(dataloader):
        if dtype == 'tensor':
            return samples[0]
        if dtype == 'list':
            inputs = samples[0].unsqueeze(0)
            return [item for item in inputs]


def rescale_tensor(tensor):
    b, c, h, w = tensor.shape
    flat_tensor = tensor.view(b, -1)
    flat_tensor -= flat_tensor.min(1, keepdim=True)[0]
    flat_tensor /= flat_tensor.max(1, keepdim=True)[0]
    tensor = flat_tensor.view(b, c, h, w)
    return tensor.permute(0, 2, 3, 1).numpy()
