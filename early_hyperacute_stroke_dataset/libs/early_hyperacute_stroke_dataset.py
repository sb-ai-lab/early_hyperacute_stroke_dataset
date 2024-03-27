import json
from pathlib import Path
from random import shuffle
from typing import Tuple, NamedTuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from early_hyperacute_stroke_dataset import conf
from early_hyperacute_stroke_dataset.libs.dataset_metadata import DatasetMetadata
from early_hyperacute_stroke_dataset.libs import normalization
from early_hyperacute_stroke_dataset.libs.augmentations import Augmentations


class DatasetItem(NamedTuple):
    study_uid: str
    slice_num: int

    slice_loc: float

    image_path: Path
    mask_path: Path
    metadata_path: Path
    raw_slice_path: Path

    metadata:  Dict[str, Any]


class EarlyHyperacuteStrokeDataset(Dataset):
    def __init__(self, dataset_path: Path, part: str, normalization_type: str, augmentations: Optional[Dict] = None) -> None:
        super().__init__()

        self._dataset_path = dataset_path
        self._part = part
        self._normalization_type = normalization.NormalizationType(normalization_type)

        self._path = self._dataset_path / self._part

        self._dataset_metadata = DatasetMetadata(dataset_path / "metadata.json")
        self._normalization_params = normalization.get_normalization_params(
            normalization_type=self._normalization_type,
            dataset_metadata=self._dataset_metadata
        )
        self._normalization_func = normalization.normalization(self._normalization_type)

        self._augmentations = Augmentations(augmentations)

        self._index_dataset()

    def __len__(self) -> int:
        return len(self._index)
    
    def __getitem__(self, i: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        item = self._index[i]

        image = np.load(item.image_path)["image"]
        mask = np.load(item.mask_path)["mask"]

        image = self._normalization_func(image, **self._normalization_params)

        if self._part == "train":
            image, mask = self._augmentations.apply(image, mask)

        image = image[np.newaxis, ...].astype(np.float32)
        input_tensor = torch.from_numpy(image)

        mask = np.stack([mask == label_num for label_num in range(len(conf.LABELS))], axis=0)
        target_tensor = torch.from_numpy(mask.astype(np.float32))

        return input_tensor, target_tensor

    def _index_dataset(self):
        self._index = list()

        for study_path in self._path.iterdir():
            if not study_path.is_dir():
                continue

            for slice_path in study_path.iterdir():
                if not slice_path.is_dir():
                    continue

                slice_metadata = self._read_slice_metadata(slice_path / "metadata.json")

                item = DatasetItem(
                    study_uid=study_path.name,
                    slice_num=int(slice_path.name),
                    slice_loc=slice_metadata["slice_location"],
                    image_path=slice_path / "image.npz",
                    mask_path=slice_path / "mask.npz",
                    metadata_path=slice_path / "metadata.json",
                    raw_slice_path=slice_path / "raw.dcm",
                    metadata=slice_metadata
                )

                self._index.append(item)
        
        shuffle(self._index)

    @staticmethod
    def _read_slice_metadata(filename: Path):
        with open(filename, "r") as fp:
            return json.load(fp)
