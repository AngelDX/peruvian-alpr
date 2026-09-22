# -*- coding: utf-8 -*-
"""Dataset PyTorch de reconocimiento de placas (crop -> secuencia de chars)."""
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from bench_utils import encode, norm_plate


class PlateDataset(Dataset):
    def __init__(self, tsv, img_h, img_w, augment=False):
        self.items = []
        with open(tsv, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                path, plate = line.split("\t")
                p = norm_plate(plate)
                if p:
                    self.items.append((path, p))
        self.h, self.w, self.augment = img_h, img_w, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, plate = self.items[i]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.h, self.w, 3), np.uint8)
        img = cv2.resize(img, (self.w, self.h))
        if self.augment:
            if np.random.rand() < 0.3:
                img = cv2.convertScaleAbs(img, alpha=np.random.uniform(0.7, 1.3),
                                          beta=np.random.uniform(-20, 20))
        img = img.astype(np.float32) / 127.5 - 1.0          # [-1,1]
        img = torch.from_numpy(img.transpose(2, 0, 1))       # CHW
        target = torch.tensor(encode(plate), dtype=torch.long)
        return img, target, plate


def collate_ctc(batch):
    imgs, targets, plates = zip(*batch)
    imgs = torch.stack(imgs, 0)
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets, 0)
    return imgs, targets, lengths, list(plates)
