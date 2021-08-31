import os
import csv
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
from abc import ABCMeta, abstractmethod

from PIL import Image
from copy import deepcopy

import torch
import torchvision
from torch.utils.data import Dataset

import transformers
from transformers.feature_extraction_utils import BatchFeature

from .base import DatasetBase


Compose = torchvision.transforms.transforms.Compose
FeatureExtractor = transformers.feature_extraction_utils.FeatureExtractionMixin


class FaceMaskDataset(DatasetBase):

    id2mask = {0: "Wear", 1: "Incorrect", 2: "Not Wear"}
    id2gender = {0: "Male", 1: "Female"}
    id2age = {0: "<30", 1: ">=30 and <60", 2: ">=60"}

    def __init__(
        self,
        data_dir: str,
        total_imgs: List[str],
        labels: List[Union[int, List[int]]],
        transform: Union[Compose, FeatureExtractor],
        return_image: bool = True,
        level: int = 1,
        is_train: bool = False,
    ):
        assert level in [1, 3]
        self.level = level
        self.data_dir = data_dir
        self.transform = transform
        self.total_imgs = total_imgs
        self.labels = labels
        self.return_image = return_image
        self.is_train = is_train

    @classmethod
    def load(cls, data_dir_or_file: str, is_train: bool = False, **kwargs):
        is_valid = kwargs.pop("is_valid", None)
        if data_dir_or_file.endswith("csv"):
            df = pd.read_csv(data_dir_or_file)
            if is_valid:
                train_df = df.groupby("is_valid").get_group(0)
                valid_df = df.groupby("is_valid").get_group(1)
                return (
                    cls(**cls.from_dataframe(train_df), is_train=is_train, **kwargs),
                    cls(**cls.from_dataframe(valid_df), is_train=is_train, **kwargs),
                )
            out = cls.from_dataframe(df)
        elif os.path.isdir(data_dir_or_file):
            out = cls.from_dir(data_dir_or_file)

        return cls(**out, is_train=is_train, **kwargs)

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        total_imgs = df["path"].tolist()
        mask = df["mask"].tolist()
        gender = df["gender"].tolist()
        age = df["age"].tolist()
        labels = list(zip(mask, gender, age))
        return {
            "total_imgs": total_imgs,
            "labels": labels,
            "data_dir": "",
        }

    @staticmethod
    def from_dir(data_dir: str):
        total_imgs = []
        for img in os.listdir(data_dir):
            if not img.startswith("._"):
                total_imgs.append(img)
        return {
            "total_imgs": total_imgs,
            "labels": [],
            "data_dir": data_dir,
        }

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.data_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        pixel_values = self.transform(image)
        if isinstance(pixel_values, BatchFeature):
            pixel_values = pixel_values["pixel_values"][0]
        if isinstance(pixel_values, np.ndarray):
            pixel_values = torch.from_numpy(pixel_values)
        output = {"pixel_values": pixel_values}
        if self.is_train:
            if self.level == 3:
                mask = self.labels[idx][0]
                gender = self.labels[idx][1]
                age = self.labels[idx][2]
                label = [mask, gender, age]
            if self.level == 1:
                try:
                    mask = self.labels[idx][0]
                    gender = self.labels[idx][1]
                    age = self.labels[idx][2]
                    label = 6 * mask + 3 * gender + age
                except (TypeError, IndexError):
                    label = self.labels[idx]
            output.update({"label": label})
        if self.return_image:
            output.update({"image": Image.open(img_loc).convert("RGB")})
        return output
