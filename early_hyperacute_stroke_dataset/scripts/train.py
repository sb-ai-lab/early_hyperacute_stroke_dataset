import argparse
import copy
from pathlib import Path
from typing import Optional, Dict, List

import optuna
import mlflow
import lightning as L
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import MLFlowLogger, Logger
from torch.utils.data import DataLoader

from early_hyperacute_stroke_dataset import conf
from early_hyperacute_stroke_dataset.libs.settings import Settings
from early_hyperacute_stroke_dataset.libs.early_hyperacute_stroke_dataset import EarlyHyperacuteStrokeDataset
from early_hyperacute_stroke_dataset.libs.segmentation_module import SegmentationModule


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path, help="path to file with training settings")

    return parser.parse_args()


def main():
    L.seed_everything(conf.RANDOM_STATE)

    args = parse_command_prompt()

    settings = Settings(args.settings)

    use_optuna_flag = settings.optuna.get("use") if settings.optuna is not None else None
    if use_optuna_flag:
        objective = Objective(settings, args.settings)
        pruner = get_pruner(settings)

        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, **settings.optuna["options"]["optimize"])

        print_optimization_result(study)
    else:
        objective = Objective(settings, args.settings)
        objective()

    print("Done.")


class Objective:
    def __init__(self, settings: Settings, settings_filename: Path):
        self._settings = settings
        self._settings_filename = settings_filename

        train_dataset = EarlyHyperacuteStrokeDataset(
            dataset_path=settings.common["dataset"],
            part="train",
            **settings.dataset
        )
        val_dataset = EarlyHyperacuteStrokeDataset(
            dataset_path=settings.common["dataset"],
            part="val",
            **settings.dataset
        )
        test_dataset = EarlyHyperacuteStrokeDataset(
            dataset_path=settings.common["dataset"],
            part="test",
            **settings.dataset
        )

        self._train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self._settings.data_loader["train_batch_size"],
            num_workers=self._settings.data_loader["num_workers"],
            shuffle=True
        )
        self._val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self._settings.data_loader["val_batch_size"],
            num_workers=self._settings.data_loader["num_workers"]
        )
        self._test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self._settings.data_loader["test_batch_size"],
            num_workers=self._settings.data_loader["num_workers"]
        )

    def __call__(self, trial: Optional[optuna.trial.Trial] = None) -> float:
        L.seed_everything(conf.RANDOM_STATE)

        if trial is not None:
            self._settings.optimizer["lr"] = trial.suggest_float(
                "lr",
                self._settings.optuna["hparams"]["optimizer"]["lr"]["min"],
                self._settings.optuna["hparams"]["optimizer"]["lr"]["max"],
                log=True
            )
        

        module = SegmentationModule(
            network_params=self._settings.network,
            loss_params=self._settings.loss,
            optimizer_params=self._settings.optimizer,
            scheduler_params=self._settings.scheduler
        )

        callbacks = Callbacks(self._settings.callbacks)
        loggers = Loggers(self._settings.loggers)

        trainer = L.Trainer(callbacks=callbacks.as_list(), logger=loggers.as_list(), **self._settings.trainer)
        trainer.fit(model=module, train_dataloaders=self._train_loader, val_dataloaders=self._val_loader)
        trainer.test(dataloaders=self._test_loader, ckpt_path="best")

        checkpoints_path = get_checkpoints_path(trainer)
        save_artifacts(loggers.mlflow, checkpoints_path, self._settings_filename)

        return trainer.checkpoint_callback.best_model_score.item()



class Callbacks:
    def __init__(self, settings: Dict):
        self.model_checkpoint = callbacks.ModelCheckpoint(**settings["model_checkpoint"])
        self._learning_rate_monitor = callbacks.LearningRateMonitor(**settings["learning_rate_monitor"])
        self._early_stopping = callbacks.EarlyStopping(**settings["early_stopping"])

    def as_list(self) -> List[callbacks.Callback]:
        return [
            self.model_checkpoint, 
            self._learning_rate_monitor,
            self._early_stopping
        ]


class Loggers:
    def __init__(self, settings: Dict):
        mlflow.set_tracking_uri(settings["mlflow"]["tracking_uri"])

        self.mlflow = MLFlowLogger(**settings["mlflow"])

    def as_list(self) -> List[Logger]:
        return [self.mlflow]



def get_pruner(settings: Settings) -> Optional[optuna.pruners.BasePruner]:
    pruner_type = settings.optuna["options"]["pruner"]["type"]

    pruner_opts = copy.deepcopy(settings.optuna["options"]["pruner"])
    del pruner_opts["type"]

    if pruner_type == "median":
        pruner = optuna.pruners.MedianPruner(**pruner_opts)
    else:
        raise NotImplementedError()

    return pruner


def print_optimization_result(study: optuna.Study) -> None:
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def get_checkpoints_path(trainer: L.Trainer) -> Path:
    return Path(trainer.checkpoint_callback.best_model_path).parent


def save_artifacts(logger: MLFlowLogger, checkpoints_path: Path, settings_filename: Path):
    mlflow_client = logger.experiment
    run_id = logger.run_id

    # Save checkpoints.
    for model_path in checkpoints_path.glob("*.ckpt"):
        mlflow_client.log_artifact(run_id, model_path, "checkpoints")

    mlflow_client.log_artifact(run_id, settings_filename)


if __name__ == "__main__":
    main()
