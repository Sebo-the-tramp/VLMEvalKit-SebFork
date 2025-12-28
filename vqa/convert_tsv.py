#!/usr/bin/env python3

import argparse
import re
import csv
import json
from pathlib import Path

from vlmeval.smp.vlm import encode_image_file_to_base64

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a JSON list of questions into the minimal TSV format."
    )
    parser.add_argument("input_json", help="Path to the source JSON file.")
    parser.add_argument("output_tsv", help="Where the TSV file will be written.")
    parser.add_argument("--mode", help="Mode: image-only or general")
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional directory to prepend when image paths in JSON are relative.",
    )
    return parser.parse_args()


def resolve_image_path(raw_path, json_dir, image_root):
    """Return a readable path for raw_path using the provided fallbacks."""
    path = Path(raw_path)
    if path.exists():
        return path

    if image_root is not None:
        candidate = image_root / raw_path
        if candidate.exists():
            return candidate

    if not path.is_absolute():
        candidate = json_dir / path
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found: {raw_path}")

def main():
    args = parse_args()
    json_path_question = Path(args.input_json).expanduser().resolve()
    json_path_answers = Path(args.input_json.replace("test_", "val_answer_", 1)).expanduser().resolve()

    image_root = (
        Path(args.image_root).expanduser().resolve()
        if args.image_root is not None
        else None
    )
    if image_root is not None and not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    with open(json_path_question, "r", encoding="utf-8") as handle:
        records = json.load(handle)
        records = [record for record in records if record["mode"] == args.mode]
        print(f"Loaded {len(records)} records from {json_path_question}")

    # print(records)

    with open(json_path_answers, "r", encoding="utf-8") as handle:
        answers_data = json.load(handle)
        answers_dict = {item["idx"]: item for item in answers_data}

    for record in records:
        idx = record["idx"]
        answer_record = answers_dict.get(idx)
        if answer_record is None:
            raise ValueError(f"No answer found for question index {idx}")
        record["answer"] = answer_record["answer"]

    with open(args.output_tsv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["index", "image_path", "question", "A", "B", "C", "D", "answer", "category"])

        for i, row in enumerate(records):
            images = row.get("file_name") or []
            if isinstance(images, str):
                images = [images]

            text = row["question"]

            # 1. Extract the Question
            # We look behind for images and look ahead for the start of the options (Newline + Letter + Dot)
            question_pattern = r"(?:<image>)+\s*(.*?)(?=\n[A-Z]\.)"
            question_match = re.search(question_pattern, text, re.DOTALL)
            question = question_match.group(1).strip() if question_match else "No question found"

            # 2. Extract the Choices
            # We find all occurrences of "Letter. Content"
            # The lookahead (?=\n[A-Z]\.|$|$) ensures we stop at the next option or end of string
            choices_pattern = r"([A-Z])\.\s+(.*?)(?=\n[A-Z]\.|$)"
            choices_raw = re.findall(choices_pattern, text)
            choices = {key: value for key, value in choices_raw}

            writer.writerow(
                [
                    i,
                    images,
                    question,
                    choices['A'],
                    choices.get('B', ''),
                    choices.get('C', ''),
                    choices.get('D', ''), # small fix for now
                    row["answer"],
                    row["category"],
                ]
            )

        print(f"Wrote {i + 1} / {len(records)} records to {args.output_tsv}", end="\r")
        print()

    print(f"Finished writing records to {args.output_tsv}")

if __name__ == "__main__":
    main()


# python convert_tsv.py /data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/seed_00/test_seed_00.json ./test.tsv --image-root /data0/sebastian.cavada/datasets/simulations_v3/dl3dv --mode general

# python convert_tsv.py /data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/seed_00/test_seed_00.json ./test_general.tsv --image-root /data0/sebastian.cavada/datasets/simulations_v3/dl3dv --mode general
