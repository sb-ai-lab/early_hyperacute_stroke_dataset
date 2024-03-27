import re
from typing import Optional, List, Tuple

import numpy as np
import pydicom as pdm


_coords_pattern = re.compile(r"(\d{1,4}.\d+), (\d{1,4}.\d+)}")


def parse_rois(ds: pdm.FileDataset) -> Optional[List[np.ndarray]]:
    if "EncapsulatedDocument" not in ds:
        return None
    
    specific_character_set = ("UTF 8", )
    encapsulated_document = pdm.charset.decode_bytes(value=ds.EncapsulatedDocument, encodings=specific_character_set, delimiters=set())

    decoded_roi = encapsulated_document.split("{")

    contours = list()
    new_contour = True
    for roi_item in decoded_roi:
        match = re.search(_coords_pattern, roi_item)

        if match is not None:
            if new_contour:
                contours.append(list())
                
                new_contour = False
            
            contours[-1].append((float(match[1]), float(match[2])))
        else:
            append_new_contour = True

            if new_contour:
                append_new_contour = False

            new_contour = True

            if append_new_contour:
                if len(contours[-1]) > 4:
                    opencv_contour = np.array([contours[-1]], dtype=np.int32).reshape((-1, 1, 2))
                    contours[-1] = opencv_contour
                else:
                    del contours[-1]
    
    return contours
