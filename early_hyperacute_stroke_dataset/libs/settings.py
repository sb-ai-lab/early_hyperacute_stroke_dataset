from pathlib import Path
from typing import Dict

import yaml

from early_hyperacute_stroke_dataset import conf


class Settings:
    def __init__(self, filename: Path):
        file_content = self._read_settings_file(filename)

        self._init_fields(file_content)
        self._postprocessing()

    def _read_settings_file(self, filename: Path):
        with open(filename, "r") as fp:
            return yaml.load(stream=fp, Loader=yaml.FullLoader)

    def _init_fields(self, file_content: Dict) -> None:
        # hparams.
        self.network = file_content["hparams"]["network"]
        self.loss = file_content["hparams"]["loss"]
        self.optimizer = file_content["hparams"]["optimizer"]
        self.scheduler = file_content["hparams"]["scheduler"]

        # Data.
        self.dataset = file_content["data"]["dataset"]
        self.data_loader = file_content["data"]["data_loader"]

        # Optuna.
        self.optuna = file_content["optuna"]

        # Training.
        self.trainer = file_content["training"]["trainer"]
        self.callbacks = file_content["training"]["callbacks"]
        self.loggers = file_content["training"]["loggers"]

        self.common = file_content["common"]
    
    def _postprocessing(self):
        self.trainer["default_root_dir"] = Path(self.common["experiments_dir"]) / self.common["experiment"]
        self.loggers["mlflow"]["experiment_name"] = self.common["experiment"]

        self.common["dataset"] = Path(self.common["dataset"])

        self.network["classes"] = len(conf.LABELS)
