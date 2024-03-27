import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pydicom as pdm
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing

from early_hyperacute_stroke_dataset import conf
from early_hyperacute_stroke_dataset.libs import helpers
from early_hyperacute_stroke_dataset.libs import data_tools
from early_hyperacute_stroke_dataset.libs import image_tools


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", required=True, type=Path, help="path to predict")
    parser.add_argument("--reference", required=True, type=Path, help="path to reference")
    parser.add_argument("--output", required=True, type=Path, help="output path")
    parser.add_argument("--show_reference", action="store_true", help="show reference on images")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    predict_index = data_tools.create_predict_index(args.predict)
    reference_index = data_tools.create_reference_index(args.reference)

    processing(args.output, predict_index, reference_index, args.show_reference)
    
    print("Done.")


def processing(
    output_path: Path,
    predict_index: Dict[str, data_tools.PredictIndexItem],
    reference_index: Dict[str, List[data_tools.ReferenceIndexSlice]],
    show_reference: bool
) -> None:
    for study_name, predict_item in predict_index.items():
        study_output_folder = output_path / study_name
        study_output_folder.mkdir()

        predict = np.load(predict_item.masks)["masks"]
        for num, reference_item in enumerate(reference_index[study_name]):
            ds = pdm.dcmread(reference_item.raw)
            image = apply_modality_lut(ds.pixel_array, ds)
            image = apply_windowing(image, ds)
            image = image_tools.float_image_to_grayscale(image, as_bgr=True)

            predict_mask = np.argmax(predict[num], axis=0)

            image = draw_layer(image, predict_mask, conf.LABELS["ischemic_core"], conf.MAIN_COLORS_PALETTE_BGR["red"])
            image = draw_layer(image, predict_mask, conf.LABELS["penumbra"], conf.MAIN_COLORS_PALETTE_BGR["yellow"])

            if show_reference:
                reference = np.load(reference_item.mask)["mask"]
                reference = np.stack([reference == label_num for label_num in range(len(conf.LABELS))], axis=0)
                reference = np.argmax(reference, axis=0)

                image = draw_layer(image, reference, conf.LABELS["ischemic_core"], conf.MAIN_COLORS_PALETTE_BGR["fuchsia"])
                image = draw_layer(image, reference, conf.LABELS["penumbra"], conf.MAIN_COLORS_PALETTE_BGR["lime"])

            cv2.imwrite((study_output_folder / f"{num:05}.png").as_posix(), image)
        
        print(f"Study {study_name} was processed successfully.")


def draw_layer(image: np.ndarray, mask: np.ndarray, value: int, color: Tuple[int, int, int]) -> np.ndarray:
    layer = np.zeros(image.shape, dtype=np.uint8)
    layer[mask == value] = color

    return cv2.addWeighted(image, 1.0, layer, 0.25, 0.0)
    

if __name__ == "__main__":
    main()
