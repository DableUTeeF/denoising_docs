from torch.utils.data import Dataset
import os
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image


class Data(Dataset):
    def __init__(self, path, w=64):
        self.path = path
        self.dirty = []
        self.clean = []
        for file in sorted(os.listdir(path)):
            if 'd' in file:
                self.dirty.append(file)
            else:
                self.clean.append(file)
        self.aug = transforms.Compose([
            transforms.Resize(w),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def __len__(self):
        return len(self.dirty)

    def __getitem__(self, idx):
        dirty = Image.open(os.path.join(self.path, self.dirty[idx]))
        clean = Image.open(os.path.join(self.path, self.clean[idx]))
        return self.aug(dirty), self.aug(clean)
