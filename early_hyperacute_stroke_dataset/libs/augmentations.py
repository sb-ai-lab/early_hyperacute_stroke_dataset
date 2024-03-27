import copy
from typing import Dict, Tuple, Optional, Type

import numpy as np
import albumentations as A


class Augmentations:
    def __init__(self, settings: Optional[Dict]):
        self._transform_pipeline = self._get_transform_pipeline(settings)
    
    def apply(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._transform_pipeline:
            transformed = self._transform_pipeline(image=image, mask=mask)

            return transformed["image"], transformed["mask"]
        else:
            return image, mask


    def _get_transform_pipeline(self, settings: Optional[Dict]) -> Optional[A.Compose]:
        transforms = list()

        if settings is not None and settings.get("use"):
            for transform in settings["transforms"]:
                transform_name, transform_params = list(transform.items())[0]

                if transform_params.get("use"):
                    transform_params = copy.deepcopy(transform_params)
                    del transform_params["use"]

                    transforms.append(self._get_transform_by_name(transform_name)(**transform_params))

            return A.Compose(transforms, is_check_shapes=False)

        return None
    
    def _get_transform_by_name(self, name) -> Type[A.DualTransform]:
        if name == "horizontal_flip":
            return A.HorizontalFlip
        elif name == "rotate":
            return A.Rotate
        else:
            raise NotImplementedError()
