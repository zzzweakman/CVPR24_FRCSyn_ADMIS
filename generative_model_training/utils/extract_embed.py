import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from iresnet import iresnet100

path = "../ckpt/elasticface_fr/295672backbone.pth"
ckpt = torch.load(path, map_location='cpu')
fr = iresnet100(num_features=512)
fr.load_state_dict(ckpt)
fr = fr.to("cuda:0")
fr.eval()

def load_img_paths(datadir):
    """load num_imgs many FFHQ images"""
    img_files = os.listdir(datadir)
    files = sorted([os.path.join(datadir, f_name) for f_name in img_files if f_name.endswith(".jpg") or f_name.endswith(".png")])
    return files

class ImageInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, datadir):
        self.img_paths = load_img_paths(datadir)
        print("Number of images:", len(self.img_paths))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        # print("----------------")
        # print(self.img_paths[index])
        image = Image.open(self.img_paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, os.path.basename(self.img_paths[index])

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)
        
bs = 1024
data_dir = "../../dataset/context_database/images"
save_dir = "../../dataset/context_database/elasticface_embeddings"
dataset = ImageInferenceDataset(data_dir)
loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)

from tqdm import tqdm
embs = []
with torch.no_grad():
    for i, (img_batch, filename_batch) in tqdm(enumerate(loader), total=len(loader)):
        print(filename_batch[0: 2])
        img_batch = torchvision.transforms.functional.resize(img_batch, 112)
        img_batch = img_batch.to("cuda:0")
        emb_batch = fr(img_batch).cpu().numpy()
        embs.append(emb_batch)

data = embs[0]
for index in range(1,len(embs)):
    data = np.concatenate((data, embs[index]),axis = 0)
print(data.shape)
print("norm...")
for index in range(data.shape[0]):
    data[index] = data[index] / np.linalg.norm(data[index])

print("done.")
np.save(save_dir,data)
       
