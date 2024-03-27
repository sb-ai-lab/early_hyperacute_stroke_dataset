import json
from pathlib import Path


class DatasetMetadata:
    def __init__(self, filename: Path):
        with open(filename, "r") as fp:
            self._metadata = json.load(fp)

        self._normalization_params = self._metadata["stats"]["train"]

    def __getitem__(self, item):
        return self._metadata[item]

    @property
    def normalization_params(self):
        return self._normalization_params
