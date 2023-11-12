import argparse
import glob
import json
import os

TEMPLATE = """defaults:
  - train_corpora:
{train_corpora_list_str}
  - valid_corpora:
{valid_corpora_list_str}
  - test_corpora:
{test_corpora_list_str}

source_vocab_config:
  vocab_build_params:
    vocab_size: {vocab_size}
    use_joined_data: true
    sampled_data_size: {sampled_data_size}
    sampling_temperature: 2.0


target_vocab_config:
  vocab_build_params:
    vocab_size: {vocab_size}
    use_joined_data: true
    sampled_data_size: {sampled_data_size}
    sampling_temperature: 2.0

binarization_config:
  max_examples_per_shard: 80_000_000

executor_config:
  slurm_partition: devaccel
  cluster: slurm

hydra:
  searchpath:
    - file://examples/nllb/modeling/components_conf
"""


def should_use_corpus(corpus_name, white_listed_corpora, black_listed_corpora):
    if white_listed_corpora is not None and corpus_name not in white_listed_corpora:
        return False
    if black_listed_corpora is not None and corpus_name not in black_listed_corpora:
        return False
    return True


def main(args):
    train_corpora_list = []
    valid_corpora_list = []
    test_corpora_list = []
    directions = args.directions.split(",")
    white_listed_corpora = (
        set(args.white_listed_corpora.split(","))
        if args.white_listed_corpora is not None
        else None
    )
    black_listed_corpora = (
        set(args.black_listed_corpora.split(","))
        if args.black_listed_corpora is not None
        else None
    )
    for direction in directions:
        train_corpora = glob.glob(
            os.path.join(
                os.path.dirname(__file__),
                f"../components_conf/train_corpora/{direction}/*.yaml",
            )
        )
        for train_corpus in train_corpora:
            _, corpus_yaml = os.path.split(train_corpus)
            corpus_name = corpus_yaml.replace(".yaml", "")
            if should_use_corpus(
                corpus_name, white_listed_corpora, black_listed_corpora
            ):
                train_corpora_list.append(f"    - {direction}/{corpus_name}")
        valid_corpora_list.append(f"    - {direction}/{args.valid_corpus}")
        test_corpora_list.append(f"    - {direction}/{args.test_corpus}")
    train_corpora_list_str = "\n".join(sorted(train_corpora_list))
    valid_corpora_list_str = "\n".join(sorted(valid_corpora_list))
    test_corpora_list_str = "\n".join(sorted(test_corpora_list))
    directory, file = os.path.split(args.output_file)
    os.makedirs(directory, exist_ok=True)
    with open(args.output_file, "w") as out_fi:
        print(
            TEMPLATE.format(
                train_corpora_list_str=train_corpora_list_str,
                valid_corpora_list_str=valid_corpora_list_str,
                test_corpora_list_str=test_corpora_list_str,
                vocab_size=args.vocab_size,
                sampled_data_size=args.sampled_data_size,
            ),
            file=out_fi,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-file",
    )
    parser.add_argument(
        "--directions",
    )
    parser.add_argument(
        "--white-listed-corpora",
        default=None,
    )
    parser.add_argument(
        "--black-listed-corpora",
        default=None,
    )
    parser.add_argument(
        "--valid-corpus",
        default="flores_dev",
    )
    parser.add_argument(
        "--test-corpus",
        default="flores_devtest",
    )
    parser.add_argument(
        "--vocab-size",
        default=256_000,
        help="SPM model size (number of units)",
        type=int,
    )
    parser.add_argument(
        "--sampled-data-size",
        default=10_000_000,
        help="Number of sentences to sample from training data to train SPM model",
        type=int,
    )
    args = parser.parse_args()
    main(args)
