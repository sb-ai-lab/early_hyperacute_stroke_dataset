import argparse
import shutil
import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from early_hyperacute_stroke_dataset.libs import helpers


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", required=True, type=Path, help="path to studies")
    parser.add_argument("--csv", required=True, type=Path, help="path to csv file")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    index = index_studies(args.studies)
    metadata = parse_csv(args.csv)

    add_data(args.output, index, metadata)

    print("Done.")


def index_studies(path: Path) -> Dict[str, Path]:
    index = dict()
    for study in path.iterdir():
        if not study.is_dir():
            continue

        index[study.name.lower()] = study
    
    return index


def parse_csv(filename: Path) -> Dict[str, Dict[str, Union[str, int, bool, None]]]:
    df = pd.read_csv(filename, delimiter=";")
    metadata = dict()
    
    df["id"] = df["id"].apply(str.lower)
    df[["sex", "age", "dsa", "nihss", "lethality"]] = df[["sex", "age", "dsa", "nihss", "lethality"]].fillna(
        np.nan).replace([np.nan], [None])
    df["sex"] = df["sex"].apply(lambda x: str.upper(x) if x else x)
    df["time"] = df[["time"]].fillna("0-24")

    for _, row in df.iterrows():        
        metadata[row.id] = {
            "sex": row.sex,
            "age": row.age if row.age is None else int(row.age),
            "dsa": row.dsa if row.dsa is None else bool(row.dsa),
            "nihss": row.nihss if row.nihss is None else int(row.nihss),
            "time": str(row.time),
            "lethality": row.lethality if row.lethality is None else bool(row.lethality)
        }
    
    return metadata


def add_data(
    output_path: Path,
    index: Dict[str, Path],
    metadata: Dict[str, Dict[str, Union[str, int, bool, None]]]
) -> None:
    for study_name, study_path in index.items():
        shutil.copytree(study_path, output_path / study_name)

        if study_name not in metadata:
            study_metadata = {
                "sex": None,
                "age": None,
                "dsa": None,
                "nihss": None,
                "time": "0-24",
                "lethality": None
            }
        else:
            study_metadata = metadata[study_name]

        write_metadata(output_path / study_name, study_metadata)


def write_metadata(path: Path, metadata: Dict[str, Union[str, int, bool, None]]):
    with open(path / "metadata.json", "w") as fp:
        json.dump(obj=metadata, fp=fp, indent=4)


if __name__ == "__main__":
    main()
