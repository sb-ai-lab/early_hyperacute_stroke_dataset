import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict, List, Union, Optional

import numpy as np
import cv2
import pydicom as pdm

from early_hyperacute_stroke_dataset import conf
from early_hyperacute_stroke_dataset.libs import helpers
from early_hyperacute_stroke_dataset.libs.parse_osirix_labeling import parse_rois


StudyType = List[Dict[str, Union[Path, float, str]]]
ContoursType = Optional[List[np.ndarray]]
LabelingType = Dict[str, Dict[str, Union[Path, ContoursType]]]


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="path to input file")
    parser.add_argument("output", type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_output_folder(args.output)

    study, masks, metadata = parse_study(args.input)

    save_study(args.output, study, masks, metadata)

    print("Done.")


def parse_study(path: Path) -> Tuple[StudyType, np.ndarray, Optional[Path]]:
    study, labeling, metadata = read_files(path)
    masks = create_masks(study, labeling)
    
    return study, masks, metadata


def read_files(path: Path) -> Tuple[StudyType, LabelingType, Optional[Path]]:
    study = list()
    labeling = dict()
    
    metadata = path / "metadata.json"
    if not metadata.exists():
        metadata = None

    sop_instance_uids = defaultdict(set)

    for dicom_filename in path.rglob("*.dcm"):
        ds = pdm.dcmread(dicom_filename, force=True)

        if "EncapsulatedDocument" in ds:
            labeling[str(ds.ContentSequence[0].ReferencedSOPSequence[0].ReferencedSOPInstanceUID)] = {
                "filename": dicom_filename,
                "rois": parse_rois(ds)
            }
        elif "PixelData" in ds:
            if ("SliceLocation" in ds
                    and "FilterType" in ds and ds.FilterType in ("1", "HEAD FILTER")
                    and ds.pixel_array.shape == conf.SLICE_SIZE):
                study.append({
                    "filename": dicom_filename,
                    "slice_loc": float(ds.SliceLocation),
                    "series_uid": str(ds.SeriesInstanceUID),
                    "sop_uid": str(ds.SOPInstanceUID)
                })

                sop_instance_uids[str(ds.SeriesInstanceUID)].add(str(ds.SOPInstanceUID))

    # All SOPInstanceUID only in one series.
    referenced_sop_instance_uids = set(labeling.keys())
    needed_series_uid = None
    for series_uid, uids in sop_instance_uids.items():
        if len(referenced_sop_instance_uids - uids) == 0:
            needed_series_uid = series_uid
            break

    clean_study = [slice_item for slice_item in study if slice_item["series_uid"] == needed_series_uid]
    study = clean_study

    study.sort(key=lambda item: item["slice_loc"])

    return study, labeling, metadata


def create_masks(study: StudyType, labeling: LabelingType) -> np.ndarray:
    objects = list()

    for slice_item in study:
        if slice_item["sop_uid"] not in labeling:
            objects.append(list())

            continue

        labels = list()
        for contour in labeling[slice_item["sop_uid"]]["rois"]:
            labels.append({
                "contour": contour,
                "area": cv2.contourArea(contour, oriented=False),
                "label": "not_processed"
            })
        
        labels.sort(key=lambda item: item["area"], reverse=True)

        for first_proc_num, first_label in enumerate(labels):
            if first_label["label"] != "not_processed":
                continue

            first_label_mask = np.zeros(conf.SLICE_SIZE, dtype=np.uint8)
            first_label_mask = cv2.drawContours(first_label_mask, [first_label["contour"]], 0, 1, -1)

            for second_proc_num in range(first_proc_num + 1, len(labels)):
                second_label = labels[second_proc_num]
                
                second_label_mask = np.zeros(conf.SLICE_SIZE, dtype=np.uint8)
                second_label_mask = cv2.drawContours(second_label_mask, [second_label["contour"]], 0, 1, -1)

                intersection_mask = first_label_mask + second_label_mask
                intersection_mask[intersection_mask != 2] = 0
                intersection_area = np.count_nonzero(intersection_mask)

                if intersection_area >= int(0.75 * second_label["area"]):
                    second_label["label"] = "ischemic_core"
                    
                    if first_label["label"] == "not_processed":
                        first_label["label"] = "penumbra"
            
            if first_label["label"] == "not_processed":
                first_label["label"] = "penumbra"

        objects.append(labels)

    masks = list()
    for item in objects:
        mask = np.zeros(conf.SLICE_SIZE, dtype=np.uint8)

        penumbra = [label["contour"] for label in item if label["label"] == "penumbra"]
        ischemic_core = [label["contour"] for label in item if label["label"] == "ischemic_core"]

        mask = cv2.drawContours(mask, penumbra, -1, conf.LABELS["penumbra"], -1)
        mask = cv2.drawContours(mask, ischemic_core, -1, conf.LABELS["ischemic_core"], -1)

        masks.append(mask)

    masks = np.stack(masks, axis=0)

    return masks


def save_study(path: Path, study: StudyType, masks: np.ndarray, metadata: Optional[Path]):
    if not path.exists():
        path.mkdir()
    
    if metadata:
        shutil.copy(metadata, path)

    for num, item in enumerate(study):
        slice_name = f"{num:05}.dcm"
        shutil.copy(item["filename"], path / slice_name)
    
    np.savez_compressed(path / "masks.npz", masks = masks)


if __name__ == "__main__":
    main()
