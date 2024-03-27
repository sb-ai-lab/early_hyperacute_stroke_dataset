import argparse
from pathlib import Path
from operator import itemgetter
from typing import Dict, List, Tuple, Callable

import numpy as np
import pydicom as pdm
import torch
from torch import nn
from pydicom.pixel_data_handlers.util import apply_modality_lut

from early_hyperacute_stroke_dataset.libs import helpers
from early_hyperacute_stroke_dataset.libs.settings import Settings
from early_hyperacute_stroke_dataset.libs.dataset_metadata import DatasetMetadata
from early_hyperacute_stroke_dataset.libs import normalization
from early_hyperacute_stroke_dataset.libs.network import init_network


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="path to model")
    parser.add_argument("--settings", required=True, type=Path, help="path to settings file")
    parser.add_argument("--dataset_metadata", required=True, type=Path, help="path to dataset metadata")
    parser.add_argument("--data", required=True, type=Path, help="path to input data")
    parser.add_argument("--output", required=True, type=Path, help="output path")
    parser.add_argument("--device_type", choices=("cpu", "cuda"), default="cpu", help="device type to inference")
    parser.add_argument("--device_num", type=int, default=0, help="device number")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    index = create_index(args.data)
    
    settings = Settings(args.settings)
    normalization_func, normalization_params = init_normalization(settings, args.dataset_metadata)
    device = init_device(args.device_type, args.device_num)

    model = init_model(args.model, settings, device)

    processing(args.output, model, index, normalization_func, normalization_params, device)

    print("Done.")


def create_index(path: Path) -> Dict[str, List[Path]]:
    index = dict()

    for study_path in path.iterdir():
        if not study_path.is_dir():
            continue

        dcms = list()
        for filename in study_path.rglob("*.dcm"):
            ds = pdm.dcmread(filename)

            dcms.append((filename, float(ds.SliceLocation)))

        dcms.sort(key=itemgetter(1))
        index[study_path.name] = dcms
    
    return index


def init_normalization(settings: Settings, dataset_metadata_path: Path) -> Tuple[Callable, Dict[str, float]]:
    normalization_type = normalization.NormalizationType(settings.dataset["normalization_type"])
    dataset_metadata = DatasetMetadata(dataset_metadata_path)

    normalization_params = normalization.get_normalization_params(
        normalization_type=normalization_type,
        dataset_metadata=dataset_metadata
    )
    normalization_func = normalization.normalization(normalization_type)
    
    return normalization_func, normalization_params


def init_device(device_type: str, device_num: int) -> torch.device:
    if device_type == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_num}")
    
    return device


def init_model(model_path: Path, settings: Settings, device: torch.device) -> nn.Module:    
    model = init_network(settings.network)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    return model


def processing(
    output_path: Path,
    model: nn.Module,
    index: Dict[str, List[Path]],
    normalization_func: Callable,
    normalization_params: Dict[str, float],
    device: torch.device
) -> None:
    with torch.no_grad():
        for study_name, files in index.items():
            study_output_path = output_path / study_name
            study_output_path.mkdir()
            
            masks = list()
            for file_item in files:
                ds = pdm.dcmread(file_item[0])
                image = apply_modality_lut(ds.pixel_array, ds)
                image = normalization_func(image, **normalization_params)

                image = image[np.newaxis, np.newaxis, ...].astype(np.float32)
                input_tensor = torch.from_numpy(image)
                input_tensor = input_tensor.to(device)

                pred = model(input_tensor)

                masks.append(pred.cpu().numpy().squeeze())
            
            masks = np.stack(masks, axis=0)
            np.savez_compressed(study_output_path / "masks.npz", masks=masks)
            print(f"Study {study_name} has been inferenced.")


if __name__ == "__main__":
    main()
