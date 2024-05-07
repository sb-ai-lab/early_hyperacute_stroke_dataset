import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from early_hyperacute_stroke_dataset import conf
from early_hyperacute_stroke_dataset.libs import helpers
from early_hyperacute_stroke_dataset.libs import data_tools


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path, help="path to dataset")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    train_index = create_index_dataset_part(args.dataset / "train")
    val_index = create_index_dataset_part(args.dataset / "val")
    test_index = create_index_dataset_part(args.dataset / "test")

    train_report = processing_dataset_part(train_index)
    val_report = processing_dataset_part(val_index)
    test_report = processing_dataset_part(test_index)

    save_report(args.output, train_report, "train")
    save_report(args.output, val_report, "val")
    save_report(args.output, test_report, "test")

    print("Done!")


def create_index_dataset_part(path: Path) -> Dict[str, List[data_tools.ReferenceIndexSlice]]:
    index = data_tools.create_reference_index(path)

    return dict(sorted(index.items()))


def processing_dataset_part(
        index: Dict[str, List[data_tools.ReferenceIndexSlice]]
) -> Dict[str, Union[Dict[str, int], Dict[str, Union[Dict[str, bool], Dict[int, int]]]]]:
    stats = {
        "only_background": 0,
        "background_and_penumbra": 0,
        "background_and_ischemic_core": 0,
        "ischemic_core_and_penumbra": 0
    }

    studies_report = dict()
    for study_name, slices in index.items():
        slices_report = dict()
        study_report = {key: False for key in conf.LABELS.keys()}

        for num, slice_data in enumerate(slices):
            mask = np.load(slice_data.mask)["mask"]

            pixels_report = count_pixels(mask)
            zones_report = detect_zones(pixels_report)

            update_study_report(study_report, zones_report)

            slices_report[f"{num:05}"] = {
                "zones": zones_report,
                "pixels": pixels_report
            }

        update_stats(study_report, stats)

        studies_report[study_name] = {
            "zones": study_report,
            "slices": slices_report
        }

    report = {
        "stats": stats,
        "studies": studies_report
    }

    return report


def count_pixels(array: np.ndarray) -> Dict[int, int]:
    stats = np.unique(array, return_counts=True)

    result = dict()
    for i in range(len(stats[0])):
        result[int(stats[0][i])] = int(stats[1][i])

    return result


def detect_zones(pixels_report: Dict[int, int]) -> Dict[str, bool]:
    report = dict()
    for label_num, label in conf.LABELS_INVERSE.items():
        if label_num in pixels_report:
            report[label] = True
        else:
            report[label] = False

    return report


def update_study_report(study_report: Dict[str, bool], slice_zones_report: Dict[str, bool]) -> None:
    for key in study_report.keys():
        study_report[key] = study_report[key] or slice_zones_report[key]


def update_stats(study_report: Dict[str, bool], stats: Dict[str, int]) -> None:
    if study_report["ischemic_core"] and study_report["penumbra"]:
        stats["ischemic_core_and_penumbra"] += 1
    elif study_report["ischemic_core"]:
        stats["background_and_ischemic_core"] += 1
    elif study_report["penumbra"]:
        stats["background_and_penumbra"] += 1
    else:
        stats["only_background"] += 1


def save_report(
        output_folder: Path,
        report: Dict[str, Union[Dict[str, bool], Dict[str, Union[Dict[str, bool], Dict[int, int]]]]],
        report_name: str
) -> None:
    with open(output_folder / f"{report_name}.json", "w") as fp:
        json.dump(obj=report, fp=fp, indent=4)


if __name__ == "__main__":
    main()
