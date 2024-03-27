import enum
from typing import Dict, Callable

import numpy as np

from early_hyperacute_stroke_dataset.libs.dataset_metadata import DatasetMetadata


@enum.unique
class NormalizationType(enum.Enum):
    NONE = None
    STANDARD_SCALING = "standard_scaling"
    MIN_MAX_SCALING = "min_max_scaling"


def get_normalization_params(
        normalization_type: NormalizationType,
        dataset_metadata: DatasetMetadata
) -> Dict:
    if normalization_type ==  NormalizationType.NONE:
        normalization_params = dict()
    elif normalization_type == NormalizationType.STANDARD_SCALING:
        normalization_params = {
            "mean": dataset_metadata.normalization_params["mean"],
            "std": dataset_metadata.normalization_params["std"]
        }
    elif normalization_type == NormalizationType.MIN_MAX_SCALING:
        normalization_params = {
            "min_val": dataset_metadata.normalization_params["min"],
            "max_val": dataset_metadata.normalization_params["max"]
        }
    else:
        raise NotImplementedError(f"Normalization type {normalization_type.name} doesn't exist.")

    return normalization_params


def normalization(normalization_type: NormalizationType) -> Callable:
    if normalization_type == NormalizationType.NONE:
        return normalization_type_none
    elif normalization_type == NormalizationType.STANDARD_SCALING:
        return normalization_type_standard_scaling
    elif normalization_type == NormalizationType.MIN_MAX_SCALING:
        return normalization_type_min_max_scaling
    else:
        raise NotImplementedError(f"Normalization type {normalization_type.name} doesn't exist.")


def denormalization(normalization_type: NormalizationType) -> Callable:
    if normalization_type == NormalizationType.NONE:
        return denormalization_type_none
    elif normalization_type == NormalizationType.STANDARD_SCALING:
        return denormalization_type_standard_scaling
    elif normalization_type == NormalizationType.MIN_MAX_WITH:
        return denormalization_type_min_max_scaling
    else:
        raise NotImplementedError(f"Normalization type {normalization_type.name} doesn't exist.")


def normalization_type_none(
        image: np.ndarray
) -> np.ndarray:
    return image


def denormalization_type_none(
        image: np.ndarray
) -> np.ndarray:
    return image


def normalization_type_standard_scaling(
        image: np.ndarray,
        mean: float,
        std: float
) -> np.ndarray:
    image = (image - mean) / std

    return image


def denormalization_type_standard_scaling(
        image: np.ndarray,
        mean: float,
        std: float
) -> np.ndarray:
    image = image * std + mean

    return image


def normalization_type_min_max_scaling(
        image: np.ndarray,
        min_val: np.ndarray,
        max_val: np.ndarray
) -> np.ndarray:
    image = (2.0 * image - max_val - min_val) / (max_val - min_val)

    return image


def denormalization_type_min_max_scaling(
        image: np.ndarray,
        min_val: np.ndarray,
        max_val: np.ndarray
) -> np.ndarray:
    image = (image * (max_val - min_val) + min_val + max_val) / 2.0

    return image
