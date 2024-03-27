import argparse
from pathlib import Path
from typing import Optional, Union, List, Dict

import cv2
import numpy as np
import pydicom as pdm

from early_hyperacute_stroke_dataset import conf
from early_hyperacute_stroke_dataset.libs import helpers
from early_hyperacute_stroke_dataset.libs import image_tools


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="path to parsed study")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_output_folder(args.output)

    study = read_study(args.input)
    masks = read_masks(args.input)

    processing(args.output, study, masks)

    print("Done.")


def read_study(path: Path) -> List[Dict[str, Union[Path, np.ndarray, float]]]:
    study = list()
    for filename in path.glob("*.dcm"):
        ds = pdm.dcmread(filename, force=True)
        
        image = pdm.pixel_data_handlers.apply_modality_lut(ds.pixel_array, ds)
        image = pdm.pixel_data_handlers.apply_windowing(image, ds)
        image = image_tools.float_image_to_grayscale(image, as_bgr=True)

        study.append({
            "filename": filename,
            "image": image,
            "loc": float(ds.SliceLocation)
        })
    
    study.sort(key=lambda item: item["loc"])

    return study


def read_masks(path: Path):
    return np.load(path / "masks.npz")["masks"]


def processing(
        output_path: Path,
        study: List[Dict[str, Union[Path, np.ndarray, float]]],
        masks: Optional[np.ndarray],
        show_mask: bool = True
) -> None:
    if not output_path.exists():
        output_path.mkdir()

    for num, slice_item in enumerate(study):
        image = slice_item["image"]

        if show_mask:
            mask = masks[num]

            unique_values = set(np.unique(mask))
            unique_values.remove(0)

            for value in unique_values:
                if value == conf.LABELS["ischemic_core"]:
                    color = conf.MAIN_COLORS_PALETTE_BGR["red"]
                else:
                    color = conf.MAIN_COLORS_PALETTE_BGR["yellow"]

                layer = np.zeros(image.shape, dtype=np.uint8)
                layer[mask == value] = color

                image = cv2.addWeighted(image, 1.0, layer, 0.25, 0.0)
        
        cv2.imwrite((output_path / f"{num:05}.png").as_posix(), image)


if __name__ == "__main__":
    main()
