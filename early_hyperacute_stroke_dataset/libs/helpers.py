import shutil
from pathlib import Path

from early_hyperacute_stroke_dataset import conf


def create_output_folder(path: Path):
    if path.exists():
        shutil.rmtree(path)
    
    path.mkdir(parents=True)


def create_folder_with_dialog(path: Path) -> None:
    if path.exists():
        answer = "#"
        while answer not in "yYnN":
            answer = input(f"Remove {path.as_posix()} (y/n)\n")
            if answer in "yY":
                shutil.rmtree(path)
            else:
                exit(0)

    path.mkdir(parents=True)


def create_color_generator():
    palette = list(conf.MAIN_COLORS_PALETTE_BGR.values())

    while True:
        for color in palette:
            yield color
