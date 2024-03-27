from pathlib import Path
from operator import attrgetter
from typing import Dict, List, NamedTuple


class PredictIndexItem(NamedTuple):
    masks: Path


class ReferenceIndexSlice(NamedTuple):
    image: Path
    mask: Path
    metadata: Path
    raw: Path
    pos_num: int


def create_predict_index(path: Path) -> Dict[str, PredictIndexItem]:
    index = dict()

    for study_path in path.iterdir():
        if not study_path.is_dir():
            continue

        index[study_path.name] = PredictIndexItem(masks=study_path / "masks.npz")
    
    return index


def create_reference_index(path: Path) -> Dict[str, List[ReferenceIndexSlice]]:
    index = dict()

    for study_path in path.iterdir():
        if not study_path.is_dir():
            continue
        
        slices = list()
        for slice_path in study_path.iterdir():
            if not slice_path.is_dir():
                continue

            slices.append(ReferenceIndexSlice(
                image=slice_path / "image.npz",
                mask=slice_path / "mask.npz",
                metadata=slice_path / "metadata.json",
                raw=slice_path / "raw.dcm",
                pos_num=int(slice_path.name)
            ))

        slices.sort(key=attrgetter("pos_num"))

        index[study_path.name] = slices
    
    return index
