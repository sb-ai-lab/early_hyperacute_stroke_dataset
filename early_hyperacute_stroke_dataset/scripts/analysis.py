import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from pprint import pprint
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import segmentation_models_pytorch as smp
from torchmetrics.classification import Dice, MulticlassJaccardIndex

from early_hyperacute_stroke_dataset import conf
from early_hyperacute_stroke_dataset.libs import helpers
from early_hyperacute_stroke_dataset.libs import data_tools


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", required=True, type=Path, help="path to predict")
    parser.add_argument("--reference", required=True, type=Path, help="path to reference")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    predict_index = data_tools.create_predict_index(args.predict)
    reference_index = data_tools.create_reference_index(args.reference)
    
    metrics = get_metrics(predict_index, reference_index)
    evaluations = analysis(metrics)

    write_results(args.output, metrics, evaluations)

    print("Evaluations:")
    pprint(object=evaluations, sort_dicts=False)

    print("Done.")


@dataclass
class StudyMetrics:
    dice_3d: Optional[float] = None
    dice: List[float] = field(init=False, repr=True)

    iou_3d: Optional[float] = None
    iou: List[float] = field(init=False, repr=True)

    sensitivity_3d: Optional[float] = None
    sensitivity: List[float] = field(init=False, repr=True)

    specificity_3d: Optional[float] = None
    specificity: List[float] = field(init=False, repr=True)

    npv_3d: Optional[float] = None
    npv: List[float] = field(init=False, repr=True)

    ppv_3d: Optional[float] = None
    ppv: List[float] = field(init=False, repr=True)

    def __post_init__(self):
        self.dice = list()
        self.iou = list()
        self.sensitivity = list()
        self.specificity = list()
        self.npv = list()
        self.ppv = list()


def get_metrics(
    predict_index: Dict[str, data_tools.PredictIndexItem],
    reference_index: Dict[str, List[data_tools.ReferenceIndexSlice]]
) -> Dict[str, StudyMetrics]:
    dice_fn = Dice(ignore_index=conf.LABELS["background"], average="macro", num_classes=len(conf.LABELS),
                   zero_division=1.0)
    iou_fn = MulticlassJaccardIndex(num_classes=len(conf.LABELS))
    
    metrics = dict()

    for study_name, predict_item in predict_index.items():
        predict = torch.argmax(torch.from_numpy(np.load(predict_item.masks)["masks"]), axis=1)
        reference = torch.from_numpy(create_full_study_mask(reference_index[study_name]))

        study_metrics = StudyMetrics()
        
        # Dice.
        study_metrics.dice_3d = float(dice_fn(predict, reference).item())
        study_metrics.dice = [float(dice_fn(predict[i], reference[i]).item()) for i in range(len(predict))]
        
        # IoU.
        study_metrics.iou_3d = float(iou_fn(predict, reference).item())
        study_metrics.iou = [float(iou_fn(predict[i], reference[i]).item()) for i in range(len(predict))]

        # Sensitivity.
        tp, fp, fn, tn = smp.metrics.get_stats(predict, reference, mode="multiclass", num_classes=len(conf.LABELS))

        study_metrics.sensitivity_3d = float(smp.metrics.sensitivity(tp, fp, fn, tn, reduction="macro").item())
        study_metrics.sensitivity = [
            float(smp.metrics.sensitivity(tp[i], fp[i], fn[i], tn[i], reduction="macro").item()) for i in
            range(len(predict))]

        # Specificity.
        study_metrics.specificity_3d = float(smp.metrics.specificity(tp, fp, fn, tn, reduction="macro").item())
        study_metrics.specificity = [
            float(smp.metrics.specificity(tp[i], fp[i], fn[i], tn[i], reduction="macro").item()) for i in
            range(len(predict))]

        # NPV.
        study_metrics.npv_3d = float(smp.metrics.negative_predictive_value(tp, fp, fn, tn, reduction="macro").item())
        study_metrics.npv = [
            float(smp.metrics.negative_predictive_value(tp[i], fp[i], fn[i], tn[i], reduction="macro").item()) for i in
            range(len(predict))]

        # PPV.
        study_metrics.ppv_3d = float(smp.metrics.positive_predictive_value(tp, fp, fn, tn, reduction="macro").item())
        study_metrics.ppv = [
            float(smp.metrics.positive_predictive_value(tp[i], fp[i], fn[i], tn[i], reduction="macro").item()) for i in
            range(len(predict))]

        metrics[study_name] = study_metrics

        print(
            f"Study {study_name} has been processed. DICE 3D: {study_metrics.dice_3d}. IoU 3D: {study_metrics.iou_3d}."
            f"Sensitivity 3D: {study_metrics.sensitivity_3d}. Specifity 3D: {study_metrics.specificity_3d}."
            f"NPV 3D: {study_metrics.npv_3d}. PPV 3D: {study_metrics.ppv_3d}.")
    
    return metrics


def create_full_study_mask(index: List[data_tools.ReferenceIndexSlice]) -> np.ndarray:
    masks = list()
    for item in index:
        mask = np.load(item.mask)["mask"]
        masks.append(mask)
    
    return np.stack(masks, axis=0)


def analysis(metrics: Dict[str, StudyMetrics]) -> Dict[str, Dict[str, float]]:
    dice_3d_values = [item.dice_3d for item in metrics.values()]
    dice_values = [value for item in metrics.values() for value in item.dice]

    iou_3d_values = [item.iou_3d for item in metrics.values()]
    iou_values = [value for item in metrics.values() for value in item.iou]

    sensitivity_3d_values = [item.sensitivity_3d for item in metrics.values()]
    sensitivity_values = [value for item in metrics.values() for value in item.sensitivity]

    specificity_3d_values = [item.specificity_3d for item in metrics.values()]
    specificity_values = [value for item in metrics.values() for value in item.specificity]

    npv_3d_values = [item.npv_3d for item in metrics.values()]
    npv_values = [value for item in metrics.values() for value in item.npv]

    ppv_3d_values = [item.ppv_3d for item in metrics.values()]
    ppv_values = [value for item in metrics.values() for value in item.ppv]

    evaluations = {
        "dice_3d": {
            "mean": float(np.mean(dice_3d_values)),
            "std": float(np.std(dice_3d_values)),
            "min": float(np.min(dice_3d_values)),
            "max": float(np.max(dice_3d_values))
        },
        "dice": {
            "mean": float(np.mean(dice_values)),
            "std": float(np.std(dice_values)),
            "min": float(np.min(dice_values)),
            "max": float(np.max(dice_values))
        },
        "iou_3d": {
            "mean": float(np.mean(iou_3d_values)),
            "std": float(np.std(iou_3d_values)),
            "min": float(np.min(iou_3d_values)),
            "max": float(np.max(iou_3d_values))
        },
        "iou": {
            "mean": float(np.mean(iou_values)),
            "std": float(np.std(iou_values)),
            "min": float(np.min(iou_values)),
            "max": float(np.max(iou_values))
        },
        "sensitivity_3d": {
            "mean": float(np.mean(sensitivity_3d_values)),
            "std": float(np.std(sensitivity_3d_values)),
            "min": float(np.min(sensitivity_3d_values)),
            "max": float(np.max(sensitivity_3d_values))
        },
        "sensitivity": {
            "mean": float(np.mean(sensitivity_values)),
            "std": float(np.std(sensitivity_values)),
            "min": float(np.min(sensitivity_values)),
            "max": float(np.max(sensitivity_values))
        },
        "specificity_3d": {
            "mean": float(np.mean(specificity_3d_values)),
            "std": float(np.std(specificity_3d_values)),
            "min": float(np.min(specificity_3d_values)),
            "max": float(np.max(specificity_3d_values))
        },
        "specificity": {
            "mean": float(np.mean(specificity_values)),
            "std": float(np.std(specificity_values)),
            "min": float(np.min(specificity_values)),
            "max": float(np.max(specificity_values))
        },
        "npv_3d": {
            "mean": float(np.mean(npv_3d_values)),
            "std": float(np.std(npv_3d_values)),
            "min": float(np.min(npv_3d_values)),
            "max": float(np.max(npv_3d_values))
        },
        "npv": {
            "mean": float(np.mean(npv_values)),
            "std": float(np.std(npv_values)),
            "min": float(np.min(npv_values)),
            "max": float(np.max(npv_values))
        },
        "ppv_3d": {
            "mean": float(np.mean(ppv_3d_values)),
            "std": float(np.std(ppv_3d_values)),
            "min": float(np.min(ppv_3d_values)),
            "max": float(np.max(ppv_3d_values))
        },
        "ppv": {
            "mean": float(np.mean(ppv_values)),
            "std": float(np.std(ppv_values)),
            "min": float(np.min(ppv_values)),
            "max": float(np.max(ppv_values))
        }
    }

    return evaluations


def write_results(output: Path, metrics: Dict[str, StudyMetrics], evaluations: Dict[str, Dict[str, float]]) -> None:
    metrics_to_write = {key: asdict(item) for key, item in metrics.items()}

    with open(output / "metrics.json", "w") as fp:
        json.dump(obj=metrics_to_write, fp=fp, indent=4)
    
    with open(output / "evaluations.json", "w") as fp:
        json.dump(obj=evaluations, fp=fp, indent=4)


if __name__ == "__main__":
    main()
