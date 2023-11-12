import argparse
import gzip
import sys

import cld3

LANG_MAP = {
    "he": "iw",
}


def open_file(file_path: str, is_gzip: bool):
    if is_gzip:
        return gzip.open(file_path, "rb")
    else:
        return open(file_path)


def main(args):
    with open(
        "src_lid." + args.output_prefix + f".src_lid.{args.src}", "w"
    ) as src_lid_out, open(
        "tgt_lid." + args.output_prefix + f".tgt_lid.{args.tgt}", "w"
    ) as tgt_lid_out:
        with open_file(args.input, args.is_gzip) as in_f:
            with open(args.output_prefix + f".{args.src}", "w") as src_out, open(
                args.output_prefix + f".{args.tgt}", "w"
            ) as tgt_out:
                for line in in_f:
                    line = line.decode("utf-8")
                    score, src_sent, tgt_sent = line.split("\t")
                    score = float(score)
                    src_sent = src_sent.strip()
                    tgt_sent = tgt_sent.strip()
                    if score >= args.threshold:
                        if args.skip_lid:
                            print(src_sent, file=src_out)
                            print(tgt_sent, file=tgt_out)
                        else:
                            src_p = cld3.get_language(src_sent)
                            if (
                                src_p.language == LANG_MAP.get(args.src, args.src)
                                and src_p.is_reliable
                            ):
                                tgt_p = cld3.get_language(tgt_sent)
                                if (
                                    tgt_p.language == LANG_MAP.get(args.tgt, args.tgt)
                                    and tgt_p.is_reliable
                                ):
                                    print(src_sent, file=src_out)
                                    print(tgt_sent, file=tgt_out)

                                else:
                                    print(tgt_sent, file=tgt_lid_out)
                            else:
                                print(src_sent, file=src_lid_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--src")
    parser.add_argument("--tgt")
    parser.add_argument(
        "--threshold", type=float, help="Minimum LASER score to include"
    )
    parser.add_argument("--skip-lid", action="store_true")
    parser.add_argument("--is-gzip", action="store_true")
    parser.add_argument("--output-prefix")
    args = parser.parse_args()
    main(args)
