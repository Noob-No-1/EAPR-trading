import json, os
import numpy as np
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        X = np.load(os.path.join(root_dir, "X.npy"))  # [N,C]
        with open(os.path.join(root_dir, "splits.json"), "r") as f:
            meta = json.load(f)
        self.X = X.astype(np.float32)
        self.lin = meta["lin"]
        self.windows = meta[f"{split}_windows"]
        self.C = self.X.shape[1]
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        s,e = self.windows[idx]
        x = self.X[s:e]                  # [L,C]
        x = torch.from_numpy(x).T        # [C,L]
        return x