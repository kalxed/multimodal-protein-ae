import glob
import os.path as osp

import torch
from torch.utils.data import Dataset


class SingleModeDataset(Dataset):
    def __init__(self, fnames:list[str] = None, dir:str = None, pattern:str = "*.pt", transform=None, use_weights_only=True, device:torch.device="cpu"):
        if fnames is None:
            self.fnames = [f for f in glob.glob(osp.join(dir, pattern))]
        else:
            self.fnames = fnames
        self.transform = transform
        self.use_weights_only = use_weights_only
        self.device=torch.device(device)
    def __len__(self):
        return len(self.fnames)
    def __getitem__(self, idx):
        data = torch.load(self.fnames[idx], weights_only=self.use_weights_only, map_location=self.device)
        if self.transform is None:
            return data
        return self.transform(data)

