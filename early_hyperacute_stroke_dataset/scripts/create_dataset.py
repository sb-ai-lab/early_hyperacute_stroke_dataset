import argparse
import copy
import shutil
import json
from pathlib import Path
from operator import attrgetter
from collections import defaultdict
from pprint import pprint
from abc import ABC, abstractmethod
from typing import List, NamedTuple, Tuple, Dict, Union, Optional

import numpy as np
import pandas as pd
import pydicom as pdm
from pydicom.pixel_data_handlers.util import apply_modality_lut
from sklearn.model_selection import train_test_split

from early_hyperacute_stroke_dataset import conf
from early_hyperacute_stroke_dataset.libs import helpers


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="path to input raw data")
    parser.add_argument("--output", required=True, type=Path, help="path to write dataset")
    parser.add_argument("--val_size", type=float, default=0.085,
                        help="should be between 0.0 and 1.0 and represent the proportion of the dataset to include in "
                             "the val split")
    parser.add_argument("--test_size", type=float, default=0.085,
                        help="should be between 0.0 and 1.0 and represent the proportion of the dataset to include "
                             "in the test split")
    
    return parser.parse_args()


def main():
    args = parse_command_prompt()
    
    helpers.create_folder_with_dialog(args.output)

    index = create_index(args.input)
    train, val, test = split_dataset(index, args.val_size, args.test_size)

    stats = get_dataset_stats(train, val, test)

    write_data(args.output, train, val, test)
    write_metadata(args.output / "metadata.json", stats, args.test_size, args.val_size)
    write_summary(args.output / "summary.csv", train, val, test)

    print("Dataset stats:")
    pprint(stats)
    print()
    
    print("Done!")


class SliceData(NamedTuple):
    file_path: Path
    location: float


class IndexItem(NamedTuple):
    name: str
    slices: List[SliceData]
    labels_path: Path
    metadata_path: Path
    metadata: Dict[str, Optional[Union[str, int, bool]]]


def create_index(path: Path) -> List[IndexItem]:
    print("Indexing raw dataset.")
    print()

    index = list()

    for study_path in path.iterdir():
        if not study_path.is_dir():
            continue
        
        slices = list()
        for slice_path in study_path.glob("*.dcm"):
            ds = pdm.dcmread(slice_path)
            slice_loc = float(ds.SliceLocation)

            slices.append(SliceData(file_path=slice_path, location=slice_loc))
        
        slices.sort(key=attrgetter("location"))

        index.append(IndexItem(
            name=study_path.name,
            slices=slices,
            labels_path=study_path / "masks.npz",
            metadata_path=study_path / "metadata.json",
            metadata=get_full_study_metadata(study_path, slices)
        ))

        print(f"Study {study_path.name} has been indexed.")
    
    index.sort(key=attrgetter("name"))

    print()
    
    return index


def get_full_study_metadata(path: Path, slices: List[SliceData]) -> Dict[str, Optional[Union[str, int, bool]]]:
    metadata = read_study_metadata(path / "metadata.json")
    additional_study_metadata = extract_additional_study_metadata(slices[0].file_path)

    manufacturer = additional_study_metadata["manufacturer"]
    model = additional_study_metadata["model"]
    device = additional_study_metadata["device"]
    age = metadata["age"] if metadata["age"] is not None else additional_study_metadata["age"]
    sex = metadata["sex"] if metadata["sex"] is not None else additional_study_metadata["sex"]
    dsa = metadata["dsa"] if metadata["dsa"] is not None else None
    nihss = metadata["nihss"]
    time = metadata["time"]
    lethality = metadata["lethality"]

    return {
        "manufacturer": manufacturer if manufacturer is not None else "unknown",
        "model": model if model is not None else "unknown",
        "device": device if device is not None else "unknown",
        "age": age if age is not None else "unknown",
        "sex": sex if sex is not None else "unknown",
        "dsa": dsa if dsa is not None else "unknown",
        "nihss": nihss if nihss is not None else "unknown",
        "time": time if time is not None else "unknown",
        "lethality": lethality if time is not None else "unknown"
    }


def split_dataset(
        index: List[IndexItem],
        val_size: float,
        test_size: float
) -> Tuple[List[IndexItem], List[IndexItem], List[IndexItem]]:
    tmp, test = train_test_split(index, test_size=test_size, random_state=conf.RANDOM_STATE, shuffle=True)
    
    new_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(tmp, test_size=new_val_size, random_state=conf.RANDOM_STATE, shuffle=True)

    return train, val, test


def get_dataset_stats(train: List[IndexItem], val: List[IndexItem], test: List[IndexItem]) -> Dict:
    print("Calculation of stats.")
    print()

    stats = {
        "common": {
            "train_size_in_studies": len(train),
            "train_size_in_images": get_dataset_size_in_images(train),
            "val_size_in_studies": len(val),
            "val_size_in_images": get_dataset_size_in_images(val),
            "test_size_in_studies": len(test),
            "test_size_in_images": get_dataset_size_in_images(test)
        }
    }

    train_stats = single_pass_on_index(train, GetMinMaxMeanModule(stats["common"]["train_size_in_images"]))
    std_val = single_pass_on_index(
        train,
        GetStdModule(mean=train_stats["mean"], amount_images=stats["common"]["train_size_in_images"])
    )

    train_stats["std"] = std_val
    stats["train"] = train_stats
    
    return stats


def write_data(path: Path, train: List[IndexItem], val: List[IndexItem], test: List[IndexItem]) -> None:
    print("Saving training part of dataset.")
    write_dataset_part(path / "train", train)

    print("Saving validation part of dataset.")
    write_dataset_part(path / "val", val)

    print("Saving testing part of dataset.")
    write_dataset_part(path / "test", test)

    print()


def write_metadata(filename: Path, stats: Dict[str, float], test_size: float, val_size: float) -> None:
    metadata = {
        "generation_params": {
            "test_size": test_size,
            "val_size": val_size
        },
        "stats": stats
    }

    with open(filename, "w") as fp:
        json.dump(obj=metadata, fp=fp, indent=4)


def write_summary(filename: Path, train: List[IndexItem], val: List[IndexItem], test: List[IndexItem]) -> None:
    train_part = convert_dataset_part_to_dict(train, "train")
    val_part = convert_dataset_part_to_dict(val, "val")
    test_part = convert_dataset_part_to_dict(test, "test")

    merged = defaultdict(list)
    for part in (train_part, val_part, test_part):
        for key in part.keys():
            merged[key].extend(part[key])

    df = pd.DataFrame(merged)
    df.to_csv(filename)


def convert_dataset_part_to_dict(part: List[IndexItem], name: str) -> Dict:
    result = defaultdict(list)

    for study in part:
        result["name"].append(study.name)

        for key in study.metadata.keys():
            result[key].append(study.metadata[key])

        result["part"].append(name)

    return result


def write_dataset_part(path: Path, dataset: List[IndexItem]) -> None:
    path.mkdir()

    for study in dataset:
        item_output_path = path / study.name
        item_output_path.mkdir()

        save_study_metadata(item_output_path / "metadata.json", study.metadata)

        masks = np.load(study.labels_path.as_posix())["masks"]

        for num, slice_data in enumerate(study.slices):
            slice_output_path = item_output_path / f"{num:05}"
            slice_output_path.mkdir()

            # Slice.
            ds = pdm.dcmread(slice_data.file_path)
            image = apply_modality_lut(ds.pixel_array, ds)
            np.savez_compressed(slice_output_path / "image.npz", image=image)

            # Mask.
            mask = masks[num]
            np.savez_compressed(slice_output_path / "mask.npz", mask=mask)

            # Other.
            save_slice_metadata(slice_output_path / "metadata.json", slice_data)
            shutil.copy(slice_data.file_path, slice_output_path / "raw.dcm")


def read_study_metadata(filename: Path) -> Dict[str, Union[str, int, bool, None]]:
    with open(filename, "r") as fp:
        metadata = json.load(fp)
    
    return metadata


def extract_additional_study_metadata(dcm_filename: Path) -> Dict[str, Optional[Union[str, int]]]:
    ds = pdm.dcmread(dcm_filename)

    manufacturer = get_value_from_ds(ds, (0x0008, 0x0070))
    model = get_value_from_ds(ds, (0x0008, 0x1090))
    device = f"{manufacturer} {model}"

    sex = get_value_from_ds(ds, (0x0010, 0x0040))
    if sex is not None:
        sex = sex.upper()

    age = get_value_from_ds(ds, (0x0010, 0x1010))
    if age is not None:
        age = int(age[0: len(age) - 1])
        if age == 0:
            age = None
    
    return {
        "manufacturer": manufacturer,
        "model": model,
        "device": device,
        "sex": sex,
        "age": age
    }


def get_value_from_ds(ds: pdm.Dataset, key: Tuple[int, int]) -> Optional[str]:
    try:
        value = str(ds[key].value)
    except:
        value = None

    if value == "":
        value = None

    return value


def save_study_metadata(
    filename: Path,
    metadata: Dict[str, Optional[Union[str, int, bool]]]
) -> None:
    with open(filename, "w") as fp:
        json.dump(obj=metadata, fp=fp, indent=4)


def save_slice_metadata(filename: Path, slice_data: SliceData) -> None:
    metadata = {
        "slice_location": slice_data.location
    }

    with open(filename, "w") as fp:
        json.dump(obj=metadata, fp=fp, indent=4)


class SinglePassModuleBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def in_cycles(self, image: np.ndarray) -> None:
        pass

    @abstractmethod
    def after_cycles(self) -> None:
        pass

    @abstractmethod
    def get_result(self):
        pass


def single_pass_on_index(index: List[IndexItem], module: SinglePassModuleBase) -> Union[Dict[str, float], float]:
    for study in index:
        for slice_data in study.slices:
            ds = pdm.dcmread(slice_data.file_path)
            image = apply_modality_lut(ds.pixel_array, ds)

            module.in_cycles(image)

    module.after_cycles()

    return module.get_result()


class GetMinMaxMeanModule(SinglePassModuleBase):
    def __init__(self, amount_images: int):
        super().__init__()

        self.__amount_images = amount_images

        self.__min = 32_000.0
        self.__max = -32_000.0

        self.__cumulative_sum = 0.0
        self.__mean = 0

    def in_cycles(self, image: np.ndarray) -> None:
        self.__min = min(self.__min, np.min(image))
        self.__max = max(self.__max, np.max(image))

        self.__cumulative_sum += np.mean(image)

    def after_cycles(self) -> None:
        self.__mean = self.__cumulative_sum / self.__amount_images

    def get_result(self) -> Dict[str, float]:
        return {
            "min": self.__min,
            "max": self.__max,
            "mean": self.__mean
        }


class GetStdModule(SinglePassModuleBase):
    def __init__(self, mean: float, amount_images: int):
        super().__init__()

        self.__mean = mean
        self.__amount_images = amount_images

        self.__cumulative_sum = 0.0
        self.__std = 0.0

    def in_cycles(self, image: np.ndarray) -> None:
        self.__cumulative_sum += np.sum(np.square((image - self.__mean)) / (image.size * self.__amount_images))

    def after_cycles(self) -> None:
        self.__std = np.sqrt(self.__cumulative_sum)

    def get_result(self) -> float:
        return self.__std


def get_dataset_size_in_images(dataset: List[IndexItem]) -> int:
    return sum([len(item.slices) for item in dataset])


if __name__ == "__main__":
    main()
