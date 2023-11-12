import argparse
import glob
import json
import os
from collections import defaultdict

TEMPLATE = """values:
  {corpus_name}:
    s3_paths:
      source: N/A
      target: N/A
    local_paths:
      source: {source_path}
      target: {target_path}
"""
FLORES_DIR = "/large_experiments/nllb/mmt/flores101"


def main(args):
    files = glob.glob(f"{args.input_path}/{args.ext}/*.{args.ext}")
    split = args.split
    for fi in files:
        directory, file_name = os.path.split(fi)
        src = file_name.split(".")[0]
        if src == "eng":
            continue
        src = src.replace("-", "_")
        tgt = "eng"
        direction = f"{src}-{tgt}"
        os.makedirs(
            os.path.join(f"components_conf/{split}_corpora", direction), exist_ok=True
        )
        reverse_direction = f"{tgt}-{src}"
        os.makedirs(
            os.path.join(f"components_conf/{split}_corpora", reverse_direction),
            exist_ok=True,
        )
        src_file = fi
        tgt_file = os.path.join(directory, f"{tgt}.{args.ext}")
        corpus_name = args.corpus_name

        with open(
            os.path.join(
                f"components_conf/{split}_corpora", direction, f"{corpus_name}.yaml"
            ),
            "w",
        ) as yaml_out:
            print(
                TEMPLATE.format(
                    corpus_name=corpus_name, source_path=src_file, target_path=tgt_file
                ),
                file=yaml_out,
            )
        with open(
            os.path.join(
                f"components_conf/{split}_corpora",
                reverse_direction,
                f"{corpus_name}.yaml",
            ),
            "w",
        ) as yaml_out:
            print(
                TEMPLATE.format(
                    corpus_name=corpus_name, source_path=tgt_file, target_path=src_file
                ),
                file=yaml_out,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valid")
    parser.add_argument(
        "--corpus-name",
        default="flores_dev",
    )
    parser.add_argument(
        "--ext",
        default=f"dev",
    )
    parser.add_argument(
        "--input-path",
        default=FLORES_DIR,
    )
    args = parser.parse_args()
    main(args)
