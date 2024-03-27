import argparse
from pathlib import Path

from early_hyperacute_stroke_dataset.libs import helpers
from early_hyperacute_stroke_dataset.scripts.parse_study import parse_study, save_study


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="input path")
    parser.add_argument("--output", required= True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_output_folder(args.output)

    for folder in args.input.iterdir():
        if not folder.is_dir():
            continue

        study, masks, metadata = parse_study(folder)
        save_study(args.output / folder.name, study, masks, metadata)

        print(f"Study {folder.name} was processed successfully.")

    print("Done.")


if __name__ == "__main__":
    main()
