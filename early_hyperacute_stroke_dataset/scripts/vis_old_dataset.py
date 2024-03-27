import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union

import cv2
import numpy as np

from  early_hyperacute_stroke_dataset import conf
from early_hyperacute_stroke_dataset.libs import helpers
from early_hyperacute_stroke_dataset.libs import image_tools


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="input path")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_output_folder(args.output)

    index = create_index(args.input)
    processing(index, args.output)

    print("Done.")


def create_index(path: Path) -> Dict[int, List[Dict[str, Union[Path, int]]]]:
    index = defaultdict(list)

    for filename in path.glob("*.npz"):
        if "_" not in filename.stem:
            continue

        study_name = int(filename.stem[0: 3])
        slice_num = int(filename.stem[4: ])

        index[study_name].append({
            "filename": filename,
            "num": slice_num
        })

    for item in index.values():
        item.sort(key=lambda x: x["num"])

    return index


def processing(index: Dict[int, List[Dict[str, Union[Path, int]]]], output_path: Path):
    for study_name, slices in index.items():
        study_output_path = output_path / str(study_name)
        study_output_path.mkdir()

        for slice_info in slices:
            try:
                data = np.load(slice_info["filename"], allow_pickle=True)

                image = image_tools.float_image_to_grayscale(data["image"], as_bgr=True)

                label = np.zeros(image.shape, dtype=np.uint8)
                label[data["label"] > 0] = conf.MAIN_COLORS_PALETTE_BGR["lime"]

                visualisation = cv2.addWeighted(image, 1.0, label, 0.25, 0.0)

                cv2.imwrite((study_output_path / f"{slice_info['num']:05}.png").as_posix(), visualisation)
            except Exception as e:
                print(f"{e}")
                continue

        print(f"Study {study_name} was processed successfully.")


if __name__ == "__main__":
    main()
