import json, os
import numpy as np
from typing import Optional
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, root_dir, split="train", with_entropy: bool = False):
        X = np.load(os.path.join(root_dir, "X.npy"))  # [N,C]
        with open(os.path.join(root_dir, "splits.json"), "r") as f:
            meta = json.load(f)
        self.X = X.astype(np.float32)
        self.lin = meta["lin"]
        # prefer filtered train windows if entropy_filter is enabled
        if split == "train" and meta.get("entropy_filter", False) and "train_windows_filtered" in meta:
            self.windows = meta["train_windows_filtered"]
        else:
            self.windows = meta[f"{split}_windows"]
        self.C = self.X.shape[1]
        self.with_entropy = bool(with_entropy)
        self.E = None
        e_path = os.path.join(root_dir, "E.npy")
        if self.with_entropy and os.path.exists(e_path):
            self.E = np.load(e_path).astype(np.float32)
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        s, e = self.windows[idx]
        x = self.X[s:e]                  # [L,C]
        x = torch.from_numpy(x).T        # [C,L]
        if self.with_entropy and (self.E is not None):
            ent = self.E[e-1] if 0 <= e-1 < len(self.E) else np.nan
            return x, np.float32(ent)
        return x