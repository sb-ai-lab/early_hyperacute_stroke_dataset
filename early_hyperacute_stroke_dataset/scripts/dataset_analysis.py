import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, Union, Tuple, Dict, List

import pydicom as pdm
from pydicom import dcmread

from early_hyperacute_stroke_dataset.libs import helpers


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="path to input data")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    train_stats = processing_dataset_part(args.input / "train")
    val_stats = processing_dataset_part(args.input / "val", )
    test_stats = processing_dataset_part(args.input / "test")

    common_stats = merge_stats([train_stats, val_stats, test_stats])

    write_stats(args.output, common_stats, train_stats, val_stats, test_stats)

    print("Done.")


def processing_dataset_part(path: Path) -> Dict[str, Dict[Optional[Union[str, int]], int]]:
    stats = {
        "manufacturer": defaultdict(int),
        "model": defaultdict(int),
        "device": defaultdict(int),
        "sex": defaultdict(int),
        "age": defaultdict(int),
        "dsa": defaultdict(int),
        "nihss": defaultdict(int),
        "time": defaultdict(int),
        "lethality": defaultdict(int)
    }

    for study in path.iterdir():
        if not study.is_dir():
            continue

        metadata = read_study_metadata(study)

        stats["manufacturer"][metadata["manufacturer"]] += 1
        stats["model"][metadata["model"]] += 1
        stats["device"][metadata["device"]] += 1
        stats["sex"][metadata["sex"]] += 1
        stats["age"][metadata["age"]] += 1
        stats["dsa"][metadata["dsa"]] += 1
        stats["nihss"]["exists" if metadata["nihss"] != "unknown" else "unknown"] += 1
        stats["time"]["known" if metadata["time"] != "0-24" else "approximately"] += 1
        stats["lethality"][metadata["lethality"]] += 1

    return stats


def read_study_metadata(path: Path) -> Dict[str, Union[str, int]]:
    with open(path / "metadata.json", "r") as fp:
        metadata = json.load(fp)

    return metadata


def merge_stats(stats_to_merge: List[Dict]) -> Dict:
    stats = {key: defaultdict(int) for key in stats_to_merge[0].keys()}

    for stat in stats_to_merge:
        for param, param_stats in stat.items():
            for key, value in param_stats.items():
                stats[param][key] += value

    return stats


def write_stats(output_path: Path, common_stats: Dict, train_stats: Dict, val_stats: Dict, test_stats: Dict) -> None:
    data = {
        "common_stats": common_stats,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "test_stats": test_stats
    }

    with open(output_path / "stats.json", "w") as fp:
        json.dump(obj=data, fp=fp, indent=4)


if __name__ == "__main__":
    main()
