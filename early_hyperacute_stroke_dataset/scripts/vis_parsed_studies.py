import argparse
from pathlib import Path

from early_hyperacute_stroke_dataset.libs import helpers
from early_hyperacute_stroke_dataset.scripts.vis_parsed_study import read_study, read_masks, processing


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="input path")
    parser.add_argument("--output", required=True, type=Path, help="output path")
    parser.add_argument("--show_mask", action="store_true", help="show mask on images")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_output_folder(args.output)

    for folder in args.input.iterdir():
        if not folder.is_dir():
            continue

        study = read_study(folder)

        if args.show_mask:
            masks = read_masks(folder)
        else:
            masks = None

        processing(args.output / folder.name, study, masks, args.show_mask)

        print(f"Study {folder.name} was processed successfully.")

    print("Done.")


if __name__ == "__main__":
    main()
