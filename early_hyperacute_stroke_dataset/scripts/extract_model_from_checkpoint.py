import argparse
from pathlib import Path

import torch


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="path to checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="output filename")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    convert(args.input, args.output)

    print("Done.")


def convert(ckpt_filename: Path, output_filename: Path) -> None:
    checkpoint = torch.load(ckpt_filename, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    new_state_dict = dict()
    for key in state_dict.keys():
        if key.startswith("_network."):
            new_key = key[len("_network."):]
        else:
            new_key = key

        new_state_dict[new_key] = state_dict[key]

    torch.save(new_state_dict, output_filename)


if __name__ == "__main__":
    main()
