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
      is_gzip: true
"""

CORPUS_NAME_MAP = {
    "gv": "GlobalVoices.v2018q4",
    "os": "OpenSubtitles.v2018",
    "jw300": "JW300.v1c",
    "wikimedia": "wikimedia.v20210402",
    "tatoeba": "Tatoeba.v20210310",
    "ted20": "Ted2020.v1",
    "infopankki": "infopankki.v1",
    "biblepk": "biblepk.v20210827",
    "biblepk1-1": "biblepk.v20210827-dedup",
}


def main(args):
    direction_directories = glob.glob(f"{args.input_path}/*-*")
    white_listed_corpora = (
        set(args.white_listed_corpora.split(","))
        if args.white_listed_corpora is not None
        else None
    )
    corpus_to_directions = defaultdict(list)
    for direction_directory in direction_directories:
        _, direction = os.path.split(direction_directory)
        os.makedirs(os.path.join(args.output_path, direction), exist_ok=True)
        src, tgt = direction.split("-")
        reverse_direction = f"{tgt}-{src}"
        if args.use_reverse_direction:
            os.makedirs(
                os.path.join(args.output_path, reverse_direction), exist_ok=True
            )
        src_files = glob.glob(f"{direction_directory}/*.{src}.gz")
        for src_file in src_files:
            corpus = os.path.split(src_file)[1].split(".")[0]
            tgt_file = os.path.join(direction_directory, f"{corpus}.{tgt}.gz")
            corpus_name = CORPUS_NAME_MAP.get(corpus, corpus)
            if corpus_name not in white_listed_corpora:
                continue
            corpus_to_directions[corpus_name].append(direction)

            with open(
                os.path.join(args.output_path, direction, f"{corpus_name}.yaml"), "w"
            ) as yaml_out:
                print(
                    TEMPLATE.format(
                        corpus_name=corpus_name,
                        source_path=src_file,
                        target_path=tgt_file,
                    ),
                    file=yaml_out,
                )
            if args.use_reverse_direction:
                corpus_to_directions[corpus_name].append(reverse_direction)
                with open(
                    os.path.join(
                        args.output_path, reverse_direction, f"{corpus_name}.yaml"
                    ),
                    "w",
                ) as yaml_out:
                    print(
                        TEMPLATE.format(
                            corpus_name=corpus_name,
                            source_path=tgt_file,
                            target_path=src_file,
                        ),
                        file=yaml_out,
                    )
    with open(args.report_file, "w") as report_out:
        for corpus_name, directions in corpus_to_directions.items():
            print(f"{corpus_name}: {len(directions)} directions", file=report_out)
            directions = sorted(directions)
            for direction in directions:
                print(f"- {direction}", file=report_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="/large_experiments/nllb/mmt/data/bitexts/mtdata/corpora",
    )
    parser.add_argument(
        "--output-path",
        default="examples/nllb/modeling/components_conf/train_corpora",
    )
    parser.add_argument(
        "--white-listed-corpora",
        default=None,
    )
    parser.add_argument(
        "--use-reverse-direction",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--report-file",
        default="corpora_reports.txt",
    )
    args = parser.parse_args()
    main(args)
