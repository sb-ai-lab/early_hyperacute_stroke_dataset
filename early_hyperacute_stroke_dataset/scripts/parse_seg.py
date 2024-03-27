import argparse
from pathlib import Path
from pprint import pprint

import pydicom as pdm

from early_hyperacute_stroke_dataset.libs.parse_osirix_labeling import parse_rois


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="path to input file")
    
    return parser.parse_args()


def main():
    args = parse_command_prompt()

    ds = pdm.dcmread(args.input)
    contours = parse_rois(ds)
    
    print("Contours:")
    pprint(contours)


if __name__ == "__main__":
    main()
